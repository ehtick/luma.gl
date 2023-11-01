// luma.gl, MIT license
// Copyright (c) vis.gl contributors

import test from 'tape-promise/tape';
import {webgl2Device} from '@luma.gl/test-utils';

import {Transform} from '@luma.gl/engine';
import {colorPicking} from '@luma.gl/shadertools';

/* eslint-disable camelcase */

const TEST_DATA = {
  vertexColorData: new Float32Array([
    0,
    0,
    0,
    255,
    100,
    150,
    50,
    50,
    50,
    251,
    103,
    153, // is picked only when threshold is 5
    150,
    100,
    255,
    254.5,
    100,
    150, // is picked with default threshold (1)
    100,
    150,
    255,
    255,
    255,
    255,
    255,
    100,
    149.5 // // is picked with default threshold (1)
  ])
};

const TEST_CASES = [
  {
    highlightedObjectColor: null,
    isPicked: [0, 0, 0, 0, 0, 0, 0, 0, 0]
  },
  {
    highlightedObjectColor: [255, 255, 255],
    isPicked: [0, 0, 0, 0, 0, 0, 0, 1, 0]
  },
  {
    highlightedObjectColor: [255, 100, 150],
    isPicked: [0, 1, 0, 0, 0, 0, 0, 0, 0]
  }
];

test.skip('colorPicking#getUniforms', (t) => {
  t.deepEqual(colorPicking.getUniforms({}), {}, 'Empty input');

  t.deepEqual(
    colorPicking.getUniforms({
      isActive: true,
      highlightedObjectColor: null,
      highlightColor: [255, 0, 0]
    }),
    {
      picking_uSelectedColorValid: 0,
      picking_uHighlightColor: [1, 0, 0, 1],
      picking_uActive: true,
      picking_uAttribute: false
    }
  );

  t.deepEqual(
    colorPicking.getUniforms({
      highlightedObjectColor: [0, 0, 1],
      highlightColor: [255, 0, 0, 51]
    }),
    {
      picking_uSelectedColorValid: 1,
      picking_uSelectedColor: [0, 0, 1],
      picking_uHighlightColor: [1, 0, 0, 0.2]
    }
  );

  t.end();
});

test('colorPicking#isVertexPicked(highlightedObjectColor invalid)', (t) => {
  if (!Transform.isSupported(webgl2Device)) {
    t.comment('Transform not available, skipping tests');
    t.end();
    return;
  }

  const VS = `\
  attribute vec3 vertexColor;
  varying float isPicked;

  void main()
  {
    isPicked = float(isVertexPicked(vertexColor));
  }
  `;
  const vertexColorData = TEST_DATA.vertexColorData;

  const elementCount = vertexColorData.length / 3;
  const vertexColor = webgl2Device.createBuffer(vertexColorData);
  const isPicked = webgl2Device.createBuffer({byteLength: elementCount * 4});

  const transform = new Transform(webgl2Device, {
    sourceBuffers: {
      vertexColor
    },
    feedbackBuffers: {
      isPicked
    },
    vs: VS,
    varyings: ['isPicked'],
    modules: [colorPicking],
    elementCount
  });

  TEST_CASES.forEach((testCase) => {
    const uniforms = colorPicking.getUniforms({
      highlightedObjectColor: testCase.highlightedObjectColor
    });

    transform.run({uniforms});

    const expectedData = testCase.isPicked;
    const outData = transform.getBuffer('isPicked').getData();

    t.deepEqual(outData, expectedData, 'Vertex should correctly get picked');
  });

  t.end();
});

/* eslint-disable max-nested-callbacks */
test('colorPicking#picking_setPickingColor', (t) => {
  if (!Transform.isSupported(webgl2Device)) {
    t.comment('Transform not available, skipping tests');
    t.end();
    return;
  }
  const VS = `\
  attribute vec3 vertexColor;
  varying float rgbColorASelected;

  void main()
  {
    picking_setPickingColor(vertexColor);
    rgbColorASelected = picking_vRGBcolor_Avalid.a;
  }
  `;

  const vertexColorData = TEST_DATA.vertexColorData;

  const elementCount = vertexColorData.length / 3;
  const vertexColor = webgl2Device.createBuffer(vertexColorData);
  const rgbColorASelected = webgl2Device.createBuffer({byteLength: elementCount * 4});

  const transform = new Transform(webgl2Device, {
    sourceBuffers: {
      vertexColor
    },
    feedbackBuffers: {
      rgbColorASelected
    },
    vs: VS,
    varyings: ['rgbColorASelected'],
    modules: [colorPicking],
    elementCount
  });

  TEST_CASES.forEach((testCase) => {
    const uniforms = colorPicking.getUniforms({
      highlightedObjectColor: testCase.highlightedObjectColor,
      // @ts-expect-error
      pickingThreshold: testCase.pickingThreshold
    });

    transform.run({uniforms});

    const outData = transform.getBuffer('rgbColorASelected').getData();

    t.deepEqual(outData, testCase.isPicked, 'Vertex should correctly get picked');
  });
  t.ok(true, 'picking_setPickingColor successful');

  t.end();
});
/* eslint-enable max-nested-callbacks */
