// luma.gl, MIT license
// Copyright (c) vis.gl contributors

import type {UniformValue, RenderPipelineProps, Binding} from '@luma.gl/core';
import type {ShaderLayout} from '@luma.gl/core';
import type {RenderPass, VertexArray} from '@luma.gl/core';
import {RenderPipeline, cast, splitUniformsAndBindings, log} from '@luma.gl/core';
import {mergeShaderLayout} from '@luma.gl/core';
// import {mergeShaderLayout, getAttributeInfosFromLayouts} from '@luma.gl/core';
import {GL} from '@luma.gl/constants';

import {getShaderLayout} from '../helpers/get-shader-layout';
import {withDeviceAndGLParameters} from '../converters/device-parameters';
import {setUniform} from '../helpers/set-uniform';
// import {copyUniform, checkUniformValues} from '../../classes/uniforms';

import {WebGLDevice} from '../webgl-device';
import {WEBGLBuffer} from './webgl-buffer';
import {WEBGLShader} from './webgl-shader';
import {WEBGLFramebuffer} from './webgl-framebuffer';
import {WEBGLTexture} from './webgl-texture';
// import {WEBGLVertexArray} from './webgl-vertex-array';
import {WEBGLRenderPass} from './webgl-render-pass';
import {WEBGLTransformFeedback} from './webgl-transform-feedback';
import {getGLDrawMode} from '../helpers/webgl-topology-utils';

const LOG_PROGRAM_PERF_PRIORITY = 4;

/** Creates a new render pipeline */
export class WEBGLRenderPipeline extends RenderPipeline {
  /** The WebGL device that created this render pipeline */
  device: WebGLDevice;
  /** Handle to underlying WebGL program */
  handle: WebGLProgram;
  /** vertex shader */
  vs: WEBGLShader;
  /** fragment shader */
  fs: WEBGLShader;
  /** The layout extracted from shader by WebGL introspection APIs */
  introspectedLayout: ShaderLayout;

  /** Uniforms set on this model */
  uniforms: Record<string, UniformValue> = {};
  /** Bindings set on this model */
  bindings: Record<string, Binding> = {};
  /** WebGL varyings */
  varyings: string[] | null = null;

  _uniformCount: number = 0;
  _uniformSetters: Record<string, Function> = {}; // TODO are these used?

  constructor(device: WebGLDevice, props: RenderPipelineProps) {
    super(device, props);
    this.device = device;
    this.handle = this.props.handle || this.device.gl.createProgram();
    this.device.setSpectorMetadata(this.handle, {id: this.props.id});

    // Create shaders if needed
    this.vs = cast<WEBGLShader>(props.vs);
    this.fs = cast<WEBGLShader>(props.fs);
    // assert(this.vs.stage === 'vertex');
    // assert(this.fs.stage === 'fragment');

    // Setup varyings if supplied
    // @ts-expect-error WebGL only
    const {varyings, bufferMode = GL.SEPARATE_ATTRIBS} = props;
    if (varyings && varyings.length > 0) {
      this.varyings = varyings;
      this.device.gl.transformFeedbackVaryings(this.handle, varyings, bufferMode);
    }

    this._linkShaders();

    this.introspectedLayout = getShaderLayout(this.device.gl, this.handle);
    // Merge provided layout with introspected layout
    this.shaderLayout = mergeShaderLayout(this.introspectedLayout, props.shaderLayout);

    // WebGPU has more restrictive topology support than WebGL
    switch (this.props.topology) {
      case 'triangle-fan-webgl':
      case 'line-loop-webgl':
        log.warn(
          `Primitive topology ${this.props.topology} is deprecated and will be removed in v9.1`
        );
        break;
      default:
    }
  }

  override destroy(): void {
    if (this.handle) {
      this.device.gl.deleteProgram(this.handle);
      // this.handle = null;
      this.destroyed = true;
    }
  }

  /**
   * Bindings include: textures, samplers and uniform buffers
   * @todo needed for portable model
   */
  setBindings(bindings: Record<string, Binding>): void {
    // if (log.priority >= 2) {
    //   checkUniformValues(uniforms, this.id, this._uniformSetters);
    // }

    for (const [name, value] of Object.entries(bindings)) {
      // Accept both `xyz` and `xyzUniforms` as valid names for `xyzUniforms` uniform block
      // This convention allows shaders to name uniform blocks as `uniform appUniforms {} app;`
      // and reference them as `app` from both GLSL and JS.
      // TODO - this is rather hacky - we could also remap the name directly in the shader layout.
      const binding =
        this.shaderLayout.bindings.find(binding => binding.name === name) ||
        this.shaderLayout.bindings.find(binding => binding.name === `${name}Uniforms`);

      if (!binding) {
        const validBindings = this.shaderLayout.bindings
          .map(binding => `"${binding.name}"`)
          .join(', ');
        log.warn(
          `Unknown binding "${name}" in render pipeline "${this.id}", expected one of ${validBindings}`
        )();
        continue; // eslint-disable-line no-continue
      }
      if (!value) {
        log.warn(`Unsetting binding "${name}" in render pipeline "${this.id}"`)();
      }
      switch (binding.type) {
        case 'uniform':
          // @ts-expect-error
          if (!(value instanceof WEBGLBuffer) && !(value.buffer instanceof WEBGLBuffer)) {
            throw new Error('buffer value');
          }
          break;
        case 'texture':
          if (!(value instanceof WEBGLTexture || value instanceof WEBGLFramebuffer)) {
            throw new Error('texture value');
          }
          break;
        case 'sampler':
          log.warn(`Ignoring sampler ${name}`)();
          break;
        default:
          throw new Error(binding.type);
      }

      this.bindings[name] = value;
    }
  }

  /** This function is @deprecated, use uniform buffers */
  setUniforms(uniforms: Record<string, UniformValue>) {
    const {bindings} = splitUniformsAndBindings(uniforms);
    Object.keys(bindings).forEach(name => {
      log.warn(
        `Unsupported value "${JSON.stringify(
          bindings[name]
        )}" used in setUniforms() for key ${name}. Use setBindings() instead?`
      )();
    });
    // TODO - check against layout
    Object.assign(this.uniforms, uniforms);
  }

  /** @todo needed for portable model
   * @note The WebGL API is offers many ways to draw things
   * This function unifies those ways into a single call using common parameters with sane defaults
   */
  draw(options: {
    renderPass: RenderPass;
    /** vertex attributes */
    vertexArray: VertexArray;
    vertexCount?: number;
    indexCount?: number;
    instanceCount?: number;
    firstVertex?: number;
    firstIndex?: number;
    firstInstance?: number;
    baseVertex?: number;
    transformFeedback?: WEBGLTransformFeedback;
  }): boolean {
    // If we are using async linking, we need to wait until linking completes
    if (this.linkStatus !== 'success') {
      return false;
    }

    const {
      renderPass,
      vertexArray,
      vertexCount,
      // indexCount,
      instanceCount,
      firstVertex = 0,
      // firstIndex,
      // firstInstance,
      // baseVertex,
      transformFeedback
    } = options;

    const glDrawMode = getGLDrawMode(this.props.topology);
    const isIndexed: boolean = Boolean(vertexArray.indexBuffer);
    const glIndexType = (vertexArray.indexBuffer as WEBGLBuffer)?.glIndexType;
    const isInstanced: boolean = Number(instanceCount) > 0;

    // Avoid WebGL draw call when not rendering any data or values are incomplete
    // Note: async textures set as uniforms might still be loading.
    // Now that all uniforms have been updated, check if any texture
    // in the uniforms is not yet initialized, then we don't draw
    if (!this._areTexturesRenderable() || vertexCount === 0) {
      // (isInstanced && instanceCount === 0)
      return false;
    }

    this.device.gl.useProgram(this.handle);

    // Note: Rebinds constant attributes before each draw call
    vertexArray.bindBeforeRender(renderPass);

    if (transformFeedback) {
      transformFeedback.begin(this.props.topology);
    }

    // We have to apply bindings before every draw call since other draw calls will overwrite
    this._applyBindings();
    this._applyUniforms();

    const webglRenderPass = renderPass as WEBGLRenderPass;
    //     // TODO - Use polyfilled WebGL2RenderingContext instead of ANGLE extension
    //     if (isIndexed && isInstanced) {
    //       // ANGLE_instanced_arrays extension
    //       this.device.gl.drawElementsInstanced(
    //         drawMode,
    //         vertexCount || 0, // indexCount?
    //         indexType,
    //         firstVertex,
    //         instanceCount || 0
    //       );
    //       // } else if (isIndexed && this.device.isWebGL2 && !isNaN(start) && !isNaN(end)) {
    //       //   this.device.gldrawRangeElements(drawMode, start, end, vertexCount, indexType, offset);
    //     } else if (isIndexed) {
    //       this.device.gl.drawElements(drawMode, vertexCount || 0, indexType, firstVertex); // indexCount?
    //     } else if (isInstanced) {
    //       this.device.gl.drawArraysInstanced(
    //         drawMode,
    //         firstVertex,
    //         vertexCount || 0,
    //         instanceCount || 0
    //       );
    //     } else {
    //       this.device.gl.drawArrays(drawMode, firstVertex, vertexCount || 0);
    //     }
    //   });

    withDeviceAndGLParameters(
      this.device,
      this.props.parameters,
      webglRenderPass.glParameters,
      () => {
        if (isIndexed && isInstanced) {
          // ANGLE_instanced_arrays extension
          this.device.gl.drawElementsInstanced(
            glDrawMode,
            vertexCount || 0, // indexCount?
            glIndexType,
            firstVertex,
            instanceCount || 0
          );
          // } else if (isIndexed && this.device.isWebGL2 && !isNaN(start) && !isNaN(end)) {
          //   this.device.gldrawRangeElements(glDrawMode, start, end, vertexCount, glIndexType, offset);
        } else if (isIndexed) {
          this.device.gl.drawElements(glDrawMode, vertexCount || 0, glIndexType, firstVertex); // indexCount?
        } else if (isInstanced) {
          this.device.gl.drawArraysInstanced(
            glDrawMode,
            firstVertex,
            vertexCount || 0,
            instanceCount || 0
          );
        } else {
          this.device.gl.drawArrays(glDrawMode, firstVertex, vertexCount || 0);
        }

        if (transformFeedback) {
          transformFeedback.end();
        }
      }
    );

    vertexArray.unbindAfterRender(renderPass);

    return true;
  }

  // PRIVATE METHODS

  // setAttributes(attributes: Record<string, Buffer>): void {}
  // setBindings(bindings: Record<string, Binding>): void {}

  protected async _linkShaders() {
    const {gl} = this.device;
    gl.attachShader(this.handle, this.vs.handle);
    gl.attachShader(this.handle, this.fs.handle);
    log.time(LOG_PROGRAM_PERF_PRIORITY, `linkProgram for ${this.id}`)();
    gl.linkProgram(this.handle);
    log.timeEnd(LOG_PROGRAM_PERF_PRIORITY, `linkProgram for ${this.id}`)();

    // TODO Avoid checking program linking error in production
    if (log.level === 0) {
      // return;
    }

    if (!this.device.features.has('shader-status-async-webgl')) {
      const status = this._getLinkStatus();
      this._reportLinkStatus(status);
      return;
    }

    // async case
    log.once(1, 'RenderPipeline linking is asynchronous')();
    await this._waitForLinkComplete();
    log.info(2, `RenderPipeline ${this.id} - async linking complete: ${this.linkStatus}`)();
    const status = this._getLinkStatus();
    this._reportLinkStatus(status);
  }

  /** Report link status. First, check for shader compilation failures if linking fails */
  _reportLinkStatus(status: 'success' | 'linking' | 'validation') {
    switch (status) {
      case 'success':
        return;

      default:
        // First check for shader compilation failures if linking fails
        if (this.vs.compilationStatus === 'error') {
          this.vs.debugShader();
          throw new Error(`Error during compilation of shader ${this.vs.id}`);
        }
        if (this.fs?.compilationStatus === 'error') {
          this.vs.debugShader();
          throw new Error(`Error during compilation of shader ${this.fs.id}`);
        }
        throw new Error(`Error during ${status}: ${this.device.gl.getProgramInfoLog(this.handle)}`);
    }
  }

  /**
   * Get the shader compilation status
   * TODO - Load log even when no error reported, to catch warnings?
   * https://gamedev.stackexchange.com/questions/30429/how-to-detect-glsl-warnings
   */
  _getLinkStatus(): 'success' | 'linking' | 'validation' {
    const {gl} = this.device;
    const linked = gl.getProgramParameter(this.handle, gl.LINK_STATUS);
    if (!linked) {
      this.linkStatus = 'error';
      return 'linking';
    }

    gl.validateProgram(this.handle);
    const validated = gl.getProgramParameter(this.handle, gl.VALIDATE_STATUS);
    if (!validated) {
      this.linkStatus = 'error';
      return 'validation';
    }

    this.linkStatus = 'success';
    return 'success';
  }

  /** Use KHR_parallel_shader_compile extension if available */
  async _waitForLinkComplete(): Promise<void> {
    const waitMs = async (ms: number) => await new Promise(resolve => setTimeout(resolve, ms));
    const DELAY_MS = 10; // Shader compilation is typically quite fast (with some exceptions)

    // If status polling is not available, we can't wait for completion. Just wait a little to minimize blocking
    if (!this.device.features.has('shader-status-async-webgl')) {
      await waitMs(DELAY_MS);
      return;
    }

    const {gl} = this.device;
    for (;;) {
      const complete = gl.getProgramParameter(this.handle, GL.COMPLETION_STATUS);
      if (complete) {
        return;
      }
      await waitMs(DELAY_MS);
    }
  }

  /**
   * Checks if all texture-values uniforms are renderable (i.e. loaded)
   * Update a texture if needed (e.g. from video)
   * Note: This is currently done before every draw call
   */
  _areTexturesRenderable() {
    let texturesRenderable = true;

    for (const [, texture] of Object.entries(this.bindings)) {
      if (texture instanceof WEBGLTexture) {
        texture.update();
        texturesRenderable = texturesRenderable && texture.loaded;
      }
    }

    return texturesRenderable;
  }

  /** Apply any bindings (before each draw call) */
  _applyBindings() {
    // If we are using async linking, we need to wait until linking completes
    if (this.linkStatus !== 'success') {
      return;
    }

    const {gl} = this.device;
    gl.useProgram(this.handle);

    let textureUnit = 0;
    let uniformBufferIndex = 0;
    for (const binding of this.shaderLayout.bindings) {
      // Accept both `xyz` and `xyzUniforms` as valid names for `xyzUniforms` uniform block
      const value =
        this.bindings[binding.name] || this.bindings[binding.name.replace(/Uniforms$/, '')];
      if (!value) {
        throw new Error(`No value for binding ${binding.name} in ${this.id}`);
      }
      switch (binding.type) {
        case 'uniform':
          // Set buffer
          const {name} = binding;
          const location = gl.getUniformBlockIndex(this.handle, name);
          if ((location as GL) === GL.INVALID_INDEX) {
            throw new Error(`Invalid uniform block name ${name}`);
          }
          gl.uniformBlockBinding(this.handle, uniformBufferIndex, location);
          // console.debug(binding, location);
          if (value instanceof WEBGLBuffer) {
            gl.bindBufferBase(GL.UNIFORM_BUFFER, uniformBufferIndex, value.handle);
          } else {
            gl.bindBufferRange(
              GL.UNIFORM_BUFFER,
              uniformBufferIndex,
              // @ts-expect-error
              value.buffer.handle,
              // @ts-expect-error
              value.offset || 0,
              // @ts-expect-error
              value.size || value.buffer.byteLength - value.offset
            );
          }
          uniformBufferIndex += 1;
          break;

        case 'texture':
          if (!(value instanceof WEBGLTexture || value instanceof WEBGLFramebuffer)) {
            throw new Error('texture');
          }
          let texture: WEBGLTexture;
          if (value instanceof WEBGLTexture) {
            texture = value;
          } else if (
            value instanceof WEBGLFramebuffer &&
            value.colorAttachments[0] instanceof WEBGLTexture
          ) {
            log.warn(
              'Passing framebuffer in texture binding may be deprecated. Use fbo.colorAttachments[0] instead'
            )();
            texture = value.colorAttachments[0];
          } else {
            throw new Error('No texture');
          }

          gl.activeTexture(GL.TEXTURE0 + textureUnit);
          gl.bindTexture(texture.target, texture.handle);
          // gl.bindSampler(textureUnit, sampler.handle);
          textureUnit += 1;
          break;

        case 'sampler':
          // ignore
          break;

        case 'storage':
        case 'read-only-storage':
          throw new Error(`binding type '${binding.type}' not supported in WebGL`);
      }
    }
  }

  /**
   * Due to program sharing, uniforms need to be reset before every draw call
   * (though caching will avoid redundant WebGL calls)
   */
  _applyUniforms() {
    for (const uniformLayout of this.shaderLayout.uniforms || []) {
      const {name, location, type, textureUnit} = uniformLayout;
      const value = this.uniforms[name] ?? textureUnit;
      if (value !== undefined) {
        setUniform(this.device.gl, location, type, value);
      }
    }
  }
}
