use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct Old {
    a: i32,
}

#[derive(Serialize, Deserialize, Debug)]
struct New {
    a: i32,
    b: Option<i32>,
}

fn main() {
    let old = Old { a: 1 };
    let json = serde_json::to_string(&old).unwrap();
    println!("JSON: {}", json);
    let new: Result<New, _> = serde_json::from_str(&json);
    println!("Result: {:?}", new);
}
