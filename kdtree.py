import random
class KdTree():
    def __init__(self,dim_ranges):

        self.dim_ranges = dim_ranges
        self.num_dims = len(self.dim_ranges)

        dim_min_list = [dim_range[0] for dim_range in dim_ranges]
        dim_max_list = [dim_range[1] for dim_range in dim_ranges]
        
        self.max_depth = 0
        self.node_count = 0

        self.root_node = KdNode(
            tree=self,
            dim_min_list=dim_min_list,
            dim_max_list=dim_max_list,
            depth=0,
            )

    def add(self,x,value):
        self.root_node.add(x,value)

    def sample(self):
        return self.root_node.sample()

class KdNode():
    EMPTY = 0
    DATA = 1
    SPLIT = 2
    def __init__(self,tree,dim_min_list, dim_max_list, depth):

        # Take refernce to the root tree object
        self.tree = tree

        # Each node covers a region of space
        # These lists mark the boundries of each dimension for this node
        self.dim_min_list = dim_min_list
        self.dim_max_list = dim_max_list 

        # Current depth of this node
        self.depth = depth

        # Update the root tree max depth 
        self.tree.max_depth = max(self.tree.max_depth, self.depth)
        self.tree.node_count += 1

        # get the dimension this node will split.
        # Just roll through the dimensions from each depth
        self.dim = self.depth % len(self.dim_min_list)

        # Compute the mid point for the current dimension
        dim_min = self.dim_min_list[self.dim]
        dim_max = self.dim_max_list[self.dim]
        self.mid_point = ( dim_min + dim_max ) / 2.0
        
        # Create empty references to the left and right children nodes.
        # These are populated as data is added
        self.left_node = None
        self.right_node = None
        self.x = None
        self.value = None

        # keep a running total of the value and count of the data below this node
        self.value_sum = 0.0
        self.count = 0
 
        self.mode = KdNode.EMPTY

        self.expected_value = 0


    def add(self,x,value):

        self.value_sum += value
        self.count += 1

        self.expected_value = self.value_sum / self.count
          
        if self.mode == KdNode.EMPTY:
            # If the node is empty, then just store the data here
            self.mode = KdNode.DATA
            self.x = x
            self.value = value
            return

        if self.mode == KdNode.DATA:
            # If this node already contains a data then we need 
            # split and pass the data to our children

            self.mode = KdNode.SPLIT

            self.create_children_nodes()

            # We need to move our data down to our left or right child node
            child_node = self.get_left_or_right_child_node(self.x)
            child_node.add(self.x,self.value)
            self.x = None
            self.value = None

            # Add this new data to the the correct child node
            child_node = self.get_left_or_right_child_node(x)
            child_node.add(x,value)
            return

        if self.mode == KdNode.SPLIT:
            # Add this new data to the the correct child node
            child_node = self.get_left_or_right_child_node(x)
            child_node.add(x,value)

    def create_children_nodes(self):

        new_dim_max_list = self.dim_max_list.copy()
        new_dim_max_list[self.dim] = self.mid_point
        
        self.left_node = KdNode(
            tree=self.tree,
            dim_min_list=self.dim_min_list,
            dim_max_list=new_dim_max_list,
            depth=self.depth + 1,
            )

        new_dim_min_list = self.dim_min_list.copy()
        new_dim_min_list[self.dim] = self.mid_point

        self.right_node = KdNode(
            tree=self.tree,
            dim_min_list=new_dim_min_list,
            dim_max_list=self.dim_max_list,
            depth=self.depth + 1,
            )


    def get_left_or_right_child_node(self,x):
        # Uses the mid point of the currect dimension to 
        # return the left or right child node.
        # It will create the child node if it doesn't exists.
        
        dim_value = x[self.dim]

        if dim_value < self.mid_point:
            return self.left_node
        else:
            return self.right_node


    def sample(self):
        # If the node has no children. Just sample unifomly within its region
        if self.mode == KdNode.EMPTY or self.mode == KdNode.DATA:
            x = [random.uniform(dim_min, dim_max) for dim_min,dim_max in zip(self.dim_min_list,self.dim_max_list)]
            return x

        # If it has children. Randomly chose a child based on the relative expected value
        else:
            if self.left_node.count > 0 and self.right_node.count >0:
                # Get the expected values for choosing left or right child
                left_expected_value = self.left_node.expected_value
                right_expected_value = self.right_node.expected_value

                # Create a probability threshold based on the relative expected values
                prob = left_expected_value / (left_expected_value + right_expected_value)
            else:
                prob = 0.5
                
            # Sample a random number then recurse down the tree
            if random.random() < prob:
                return self.left_node.sample()
            else:
                return self.right_node.sample()
