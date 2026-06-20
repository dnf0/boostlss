#[derive(Debug, Clone)]
pub struct Tree {
    pub feature_names: Vec<String>,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
}

impl Tree {
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            feature_names,
            max_depth: 2,
            min_samples_leaf: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub enum TreeNode {
    Leaf(f64),
    Split {
        feature_idx: usize,
        split_val: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn scale(&mut self, nu: f64) {
        match self {
            TreeNode::Leaf(val) => *val *= nu,
            TreeNode::Split { left, right, .. } => {
                left.scale(nu);
                right.scale(nu);
            }
        }
    }
}
