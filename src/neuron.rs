use ndarray::Array1;

pub struct Neuron {
    pub inputs: Array1<f64>,
    pub weights: Array1<f64>,
    pub bias: f64,
}

pub trait NeuronOps {
    fn calculate_output(&self) -> f64;
}

impl NeuronOps for Neuron {
    fn calculate_output(&self) -> f64 {
        let dot_product: f64 = self.inputs.dot(&self.weights);
        return dot_product + &self.bias;
    }
}
