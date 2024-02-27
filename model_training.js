import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/real-estate-data.js'

const INPUTS = TRAINING_DATA.inputs

const OUTPUTS = TRAINING_DATA.outputs

tf.util.shuffleCombo(INPUTS,OUTPUTS)

const INPUT_TENSOR = tf.tensor2d(INPUTS)
// console.log(INPUT_TENSOR)

const OUTPUT_TENSOR = tf.tensor1d(OUTPUTS)
// console.log(OUTPUT_TENSOR)

const normalize = (tensor,min,max) =>{
    const result = tf.tidy(()=>{
        const MIN_VALUES = min || tf.min(tensor,0)

        const MAX_VALUES = max || tf.max(tensor,0)

        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor,MIN_VALUES)

        const RANGE_SIZE = tf.sub(MAX_VALUES,MIN_VALUES)

        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE,RANGE_SIZE)

        return {NORMALIZED_VALUES,MIN_VALUES,MAX_VALUES}
    })
    return result
}

const FEATURE_RESULTS = normalize(INPUT_TENSOR)

console.log(`Normalised Values:`)
FEATURE_RESULTS.NORMALIZED_VALUES.print()

console.log(`Min Values:`)
FEATURE_RESULTS.MIN_VALUES.print()

console.log(`Max Values:`)
FEATURE_RESULTS.MAX_VALUES.print()

INPUT_TENSOR.dispose()

const model = tf.sequential()

model.add(tf.layers.dense({inputShape: [2],units:1}))

model.summary()

const train = async ()=>{
    const LEARNING_RATE = 0.1

    model.compile({
        optimizer: tf.train.sgd(LEARNING_RATE),
        loss: 'meanSquaredError'
    })

    let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES,OUTPUT_TENSOR, {
        validationSplit: 0.15,
        shuffle: true,
        batchSize: 64,
        epoch: 10
    })

    OUTPUT_TENSOR.dispose()
    FEATURE_RESULTS.NORMALIZED_VALUES.dispose()

    console.log(`Average error loss: ${Math.sqrt(results.history.loss[results.history.loss.length - 1])}`)

    console.log(`Average validation error loss: ${Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1])}`)
}

await train()

const evaluate = ()=>{
    tf.tidy(()=>{
        let newInput = normalize(tf.tensor2d([[750,1]]),FEATURE_RESULTS.MIN_VALUES,FEATURE_RESULTS.MAX_VALUES)

        let output = model.predict(newInput.NORMALIZED_VALUES)
        output.print()
    })
}

evaluate()

await model.save('downloads://linear-regression-model')

FEATURE_RESULTS.MIN_VALUES.dispose()
FEATURE_RESULTS.MAX_VALUES.dispose()
model.dispose()

console.log(tf.memory().numTensors)