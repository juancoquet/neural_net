use ndarray::{Array1, Array2};
use rand::Rng;

pub struct LayerDense {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

impl LayerDense {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        let mut blank_weights = vec![vec![0.0; n_neurons]; n_inputs]; // pre-transposed
        let mut rng = rand::thread_rng();
        for neuron in &mut blank_weights {
            for weight in neuron {
                *weight = rng.gen_range(-1.0..1.0) * 0.01;
            }
        }

        let flattened: Vec<f64> = blank_weights.into_iter().flatten().collect();
        let weights = Array2::from_shape_vec((n_inputs, n_neurons), flattened).unwrap();
        let biases = Array1::<f64>::zeros(n_neurons);

        return LayerDense { weights, biases };
    }

    pub fn forward(self, inputs: Array2<f64>) -> Array2<f64> {
        let output = inputs.dot(&self.weights) + self.biases;
        return output;
    }
}
