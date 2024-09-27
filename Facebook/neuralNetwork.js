const tf = require('@tensorflow/tfjs');

async function trainModel(data, labels) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [data[0].length] }));
  model.add(tf.layers.dense({ units: labels[0].length, activation: 'softmax' }));
  
  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  
  await model.fit(tf.tensor2d(data), tf.tensor2d(labels), { epochs: 100 });
  return model;
}

async function predict(model, input) {
  const prediction = model.predict(tf.tensor2d([input]));
  return prediction.arraySync()[0];
}

module.exports = { trainModel, predict };