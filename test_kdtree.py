import pytest
import numpy as np
from kdtree import KdTree

def test_root_node():
    model = KdTree(dim_ranges=[
        [1,2], # dim 1 min, max
        [3,4], # dim 2 min, max
        ]) 
    assert model.root_node.dim_min_list == [1,3]
    assert model.root_node.dim_max_list == [2,4]
    assert model.root_node.mid_point == (1.+2.)/2.0
    assert model.root_node.depth == 0
    assert model.root_node.dim == 0
    


def test_2x2():
    model = KdTree(dim_ranges=[
        [0,10],
        [0,10],
        ])

    model.add([3,3],1)
    model.add([3,7],2)
    model.add([7,3],3)
    model.add([7,7],4)

    root = model.root_node
    l = root.left_node
    r = root.right_node

    ll = l.left_node
    lr = l.right_node
    rl = r.left_node
    rr = r.right_node

    assert model.max_depth == 2
    
    # Check branching nodes don't have data
    assert root.x == None
    assert root.value == None
    assert l.x == None
    assert l.value == None
    assert r.x == None
    assert r.value == None

    assert root.dim == 0
    assert root.mid_point == 5
    assert l.dim == 1
    assert l.mid_point == 5
    assert l.dim == 1
    assert l.mid_point == 5

    # Check leaf nodes hold correct data
    assert ll.x == [3,3]
    assert lr.x == [3,7]
    assert rl.x == [7,3]
    assert rr.x == [7,7]

    assert ll.value == 1
    assert lr.value == 2
    assert rl.value == 3
    assert rr.value == 4

    # Check leaf nodes have rolled back to the first dimension
    assert ll.dim == 0
    assert lr.dim == 0
    assert rl.dim == 0
    assert rr.dim == 0

    # Check all node report the correct expected value
    assert root.expected_value == (1+2+3+4) / 4.0
    assert l.expected_value == (1+2) / 2.0
    assert r.expected_value == (3+4) / 2.0
    assert ll.expected_value == 1
    assert lr.expected_value == 2
    assert rl.expected_value == 3
    assert rr.expected_value == 4

    # Check each leaf node has correct boundries
    assert ll.dim_min_list == [0,0]
    assert lr.dim_min_list == [0,5]
    assert rl.dim_min_list == [5,0]
    assert rr.dim_min_list == [5,5]

    assert ll.dim_max_list == [5,5]
    assert lr.dim_max_list == [5,10]
    assert rl.dim_max_list == [10,5]
    assert rr.dim_max_list == [10,10]



def test_sample_2x2():
    model = KdTree(dim_ranges=[
        [0,10],
        [0,10],
        ])

    model.add([3,3],1)
    model.add([3,7],2)
    model.add([7,3],3)
    model.add([7,7],4)

    root = model.root_node
    l = root.left_node
    r = root.right_node

    ll = l.left_node
    lr = l.right_node
    rl = r.left_node
    rr = r.right_node

    assert len(ll.sample()) == 2
    assert len(model.sample()) == 2
    
