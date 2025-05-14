use std::{cell::RefCell, collections::HashSet, ops::Deref, rc::Rc};

#[derive(Clone, Copy, Debug)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Div,
    Relu,
}

#[derive(Clone, Debug)]
pub struct ValueData {
    pub data: f32,
    pub grad: f32,

    pub src: Vec<Value>,
    pub op: Option<OpType>,
    pub param: bool,
}

#[derive(Clone, Debug)]
pub struct Value(Rc<RefCell<ValueData>>);

impl<T: Into<f32>> From<T> for Value {
    fn from(value: T) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data: value.into(),
            grad: 0.,
            src: vec![],
            op: None,
            param: false,
        })))
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueData>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Value {
    fn new(data: f32, grad: f32, src: Vec<Value>, op: Option<OpType>) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            data,
            grad,
            src,
            op,
            param: false,
        })))
    }

    pub fn add(&self, rhs: &Value) -> Value {
        let data = self.borrow().data + rhs.borrow().data;
        let src = vec![self.clone(), rhs.clone()];
        Value::new(data, 0., src, Some(OpType::Add))
    }

    pub fn sub(&self, rhs: &Value) -> Value {
        let data = self.borrow().data - rhs.borrow().data;
        let src = vec![self.clone(), rhs.clone()];
        Value::new(data, 0., src, Some(OpType::Sub))
    }

    pub fn mul(&self, rhs: &Value) -> Value {
        let data = self.borrow().data * rhs.borrow().data;
        let src = vec![self.clone(), rhs.clone()];
        Value::new(data, 0., src, Some(OpType::Mul))
    }

    pub fn div(&self, rhs: &Value) -> Value {
        let data = self.borrow().data / rhs.borrow().data;
        let src = vec![self.clone(), rhs.clone()];
        Value::new(data, 0., src, Some(OpType::Div))
    }

    pub fn relu(&self) -> Value {
        let mut data = self.borrow().data;
        if data < 0. {
            data = 0.0;
        }
        let src = vec![self.clone()];
        Value::new(data, 0., src, Some(OpType::Relu))
    }

    pub fn backward(&self) {
        let mut visited = HashSet::new();
        let mut topo = vec![];

        build_toop(self, &mut visited, &mut topo);
        topo.reverse();

        self.borrow_mut().grad = 1.0;
        for v in topo {
            // println!("{:#?}", v);
            let grad = v.borrow().grad;
            if let Some(op) = v.borrow().op {
                match op {
                    OpType::Add => {
                        let child0 = &v.borrow().src[0];
                        let child1 = &v.borrow().src[1];

                        child0.borrow_mut().grad += grad;
                        child1.borrow_mut().grad += grad;
                    }
                    OpType::Sub => {
                        let child0 = &v.borrow().src[0];
                        let child1 = &v.borrow().src[1];

                        child0.borrow_mut().grad += grad;
                        child1.borrow_mut().grad -= grad;
                    }
                    OpType::Mul => {
                        let child0 = &v.borrow().src[0];
                        let child1 = &v.borrow().src[1];

                        {
                            let data = child1.borrow().data;
                            child0.borrow_mut().grad += grad * data;
                        }
                        {
                            let data = child0.borrow().data;
                            child1.borrow_mut().grad += grad * data;
                        }
                    }
                    OpType::Div => {
                        let child0 = &v.borrow().src[0];
                        let child1 = &v.borrow().src[1];

                        child0.borrow_mut().grad += v.borrow().grad / child1.borrow().data;
                        child1.borrow_mut().grad -= v.borrow().grad * child0.borrow().data
                            / child1.borrow().data
                            / child1.borrow().data;
                    }
                    OpType::Relu => {
                        let child = &v.borrow().src[0];
                        child.borrow_mut().grad += if v.borrow().data > 0. { grad } else { 0.0 }
                    }
                }
            }
        }
    }
}

fn build_toop(node: &Value, visited: &mut HashSet<*const ValueData>, topo: &mut Vec<Value>) {
    let ptr = node.as_ptr() as *const ValueData;
    if visited.contains(&ptr) {
        return;
    }
    visited.insert(ptr);
    for child in &node.borrow().src {
        build_toop(child, visited, topo);
    }
    topo.push(node.clone());
}
