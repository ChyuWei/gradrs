use std::iter::once;

use crate::engine::Value;

pub trait Module {
    fn zero_grad(&self) {
        for v in self.params() {
            v.borrow_mut().grad = 0.0;
        }
    }

    fn step(&self, lr: f32) {
        for v in self.params() {
            let grad = v.borrow().grad;
            v.borrow_mut().data -= lr * grad;
        }
    }

    fn params(&self) -> Vec<Value>;
}

pub struct Neuron {
    w: Vec<Value>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: isize, nonlin: bool) -> Self {
        let mut w = vec![];
        for _ in 0..nin {
            let v: Value = rand::random_range(-1.0..=1.0).into();
            v.borrow_mut().param = true;
            w.push(v);
        }
        let b: Value = rand::random_range(-1.0..=1.0).into();
        b.borrow_mut().param = true;

        Neuron { w, b, nonlin }
    }

    pub fn call(&self, x: &[Value]) -> Value {
        let mut res = self.b.clone();
        for e in self.w.iter().zip(x.iter()).map(|(w, x)| w.mul(x)) {
            res = res.add(&e);
        }

        if self.nonlin {
            res = res.relu();
        }

        res
    }

    pub fn print(&self) {
        let ws: Vec<String> = self
            .w
            .iter()
            .map(|w| format!("{} {}", w.borrow().data, w.borrow().grad))
            .collect();

        println!(
            "w: ({}), b:({} {}) ",
            ws.join(","),
            self.b.borrow().data,
            self.b.borrow().grad
        );
    }
}

impl Module for Neuron {
    fn params(&self) -> Vec<Value> {
        self.w.iter().cloned().chain(once(self.b.clone())).collect()
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    fn new(nin: isize, nout: isize, nonlin: bool) -> Self {
        let mut neurons = vec![];
        for _ in 0..nout {
            neurons.push(Neuron::new(nin, nonlin))
        }

        Layer { neurons }
    }

    fn call(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x)).collect()
    }

    fn print(&self) {
        for n in &self.neurons {
            n.print();
        }
        println!();
    }
}

impl Module for Layer {
    fn params(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.params()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(size: Vec<(isize, isize)>) -> Self {
        MLP {
            layers: size
                .iter()
                .enumerate()
                .map(|(idx, (nin, nout))| Layer::new(*nin, *nout, idx != size.len() - 1))
                .collect(),
        }
    }

    pub fn call(&self, x: &[Value]) -> Vec<Value> {
        let mut res = Vec::from(x);
        for l in &self.layers {
            res = l.call(&res);
        }
        res
    }

    pub fn print(&self) {
        for l in &self.layers {
            l.print();
        }
    }
}

impl Module for MLP {
    fn params(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }
}
