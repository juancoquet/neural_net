pub struct Neuron {
    pub inputs: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: f64,
}

pub fn get_output(neuron: Neuron) -> f64 {
    return neuron.inputs[0] * neuron.weights[0]
        + neuron.inputs[1] * neuron.weights[1]
        + neuron.inputs[2] * neuron.weights[2]
        + neuron.inputs[3] * neuron.weights[3]
        + neuron.bias;
}
