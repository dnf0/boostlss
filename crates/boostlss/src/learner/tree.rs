use ndarray::Array1;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TreeNode {
    Leaf {
        value: f64,
        samples: usize,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

impl TreeNode {
    pub fn scale(&mut self, factor: f64) {
        match self {
            TreeNode::Leaf { value, .. } => {
                *value *= factor;
            }
            TreeNode::Split { left, right, .. } => {
                left.scale(factor);
                right.scale(factor);
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tree {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
}

impl Tree {
    pub fn new() -> Self {
        Self {
            max_depth: 3,
            min_samples_leaf: 1,
        }
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }
}
