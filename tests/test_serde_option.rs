use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Foo {
    a: i32,
    b: Option<i32>,
}

#[test]
fn test_serde() {
    let json = r#"{"a": 1}"#;
    let foo: Result<Foo, _> = serde_json::from_str(json);
    assert!(foo.is_err());
}
