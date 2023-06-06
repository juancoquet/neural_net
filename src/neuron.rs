use ndarray::Array1;

pub struct Neuron {
    pub inputs: Array1<f64>,
    pub weights: Array1<f64>,
    pub bias: f64,
}

pub fn calculate_output(neuron: Neuron) -> f64 {
    let dot_product: f64 = neuron.inputs.dot(&neuron.weights);
    return dot_product + neuron.bias;
}
