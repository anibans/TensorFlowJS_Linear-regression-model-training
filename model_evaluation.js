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

INPUT_TENSOR.dispose()

const MODEL_PATH = 'model/linear-regression-model.json'
const model = await tf.loadLayersModel(MODEL_PATH)

model.summary()


const evaluate = ()=>{
    tf.tidy(()=>{
        let newInput = normalize(tf.tensor2d([[750,1]]),FEATURE_RESULTS.MIN_VALUES,FEATURE_RESULTS.MAX_VALUES)

        let output = model.predict(newInput.NORMALIZED_VALUES)
        output.print()
    })
}

evaluate()

FEATURE_RESULTS.MIN_VALUES.dispose()
FEATURE_RESULTS.MAX_VALUES.dispose()
model.dispose()

console.log(tf.memory().numTensors)