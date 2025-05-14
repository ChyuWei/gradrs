use gradrs::{
    engine::Value,
    nn::{MLP, Module},
};

fn main() {
    let xx: Vec<Value> = (0..10).map(|x| Value::from(x as f32)).collect();
    let yy: Vec<Value> = xx.iter().map(|x| x.add(&Value::from(1.0))).collect();
    let xx: Vec<Vec<Value>> = xx.iter().map(|x| vec![x.clone()]).collect();

    let mlp = MLP::new(vec![(1, 10), (10, 1)]);

    for epoch in 0..5000 {
        let mut sum = Value::from(0.0);
        for (zz, y) in xx.iter().map(|x| mlp.call(x)).zip(yy.iter()) {
            let z = &zz[0];
            let diff = z.sub(y);
            let square = diff.mul(&diff);
            sum = sum.add(&square);
        }
        let mse = sum.mul(&Value::from(1. / xx.len() as f32));

        mlp.zero_grad();
        mse.backward();
        mlp.step(0.01);

        if epoch % 100 == 0 {
            println!("loss: {}", mse.borrow().data);
        }
    }
}
