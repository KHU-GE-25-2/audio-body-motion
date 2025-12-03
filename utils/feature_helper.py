import numpy as np 
import pyquaternion as pyq


def average(arr, n): # Replace every "n" values by their average
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def split_bvh_content(bvh_content):
    for idx, line in enumerate(bvh_content) : 
        if line.startswith('MOTION'):
            motion_idx = idx 
            break 
    hierarchy = bvh_content[: motion_idx]
    bvh_info  = bvh_content[motion_idx: motion_idx+3]
    bvh_frame = bvh_content[motion_idx+3: ]

    return hierarchy, bvh_info, bvh_frame


def create_hierarchy_nodes(hierarchy, depth_space=4):
    joint_offsets = []
    joint_names = []
    joint_level = []

    for line in hierarchy:
        line = line.replace('\t', ' ' * depth_space)
        level = int((len(line) - len(line.lstrip())) / depth_space)

        line = line.split()
        if not len(line) == 0:
            line_type = line[0]
            if line_type == 'OFFSET':
                offset = np.array([float(line[1]), float(line[2]), float(line[3])])
                joint_offsets.append(offset)
            elif line_type == 'ROOT' or line_type == 'JOINT':
                joint_names.append(line[1])
                joint_level.append(level)
            elif line_type == 'End':
                joint_names.append('End Site')
                joint_level.append(level)
    
    nodes = []
    for idx, (offset, name, level) in enumerate(zip(joint_offsets, joint_names, joint_level)): 
        cur_level = level
        parent = None
        children = []
        for back_idx in range(idx+1, len(joint_names)):
            if joint_level[back_idx] == cur_level : 
                break 
            if joint_level[back_idx] == cur_level + 1 : 
                children.append(back_idx)
        for front_idx in range(idx-1, -1, -1):
            if joint_level[front_idx] == cur_level - 1 :
                parent = int(front_idx)
                break 

        node = dict([('name', name), 
                     ('parent', parent), 
                     ('children', children), 
                     ('offset', joint_offsets[idx]), 
                     ('rel_degs', None), 
                     ('abs_qt', None), 
                     ('rel_pos', None), 
                     ('abs_pos', None)])
        
        if idx == 0:
            node['rel_pos'] = node['abs_pos'] = [float(0), float(60), float(0)]
            node['abs_qt'] = pyq.Quaternion()
        nodes.append(node)
        
    return nodes



def rot_vec_to_abs_pos_vec(frames, nodes, n_outputs):
    output_lines = []

    for frame_idx, frame in enumerate(frames):
        node_idx = 0
        assert (len(frame)//3) * 3 == len(frame), 'check the num of rotation point and joint num'
        for i in range(len(frame)//3): # 전체 joint 갯수
            stepi = i*3
            z_deg = float(frame[stepi])
            x_deg = float(frame[stepi+1])
            y_deg = float(frame[stepi+2])

            if nodes[node_idx]['name'] == 'End Site':
                node_idx = node_idx + 1
            nodes[node_idx]['rel_degs'] = [z_deg, x_deg, y_deg]
            current_node = nodes[node_idx]

            node_idx = node_idx + 1

        for start_node in nodes:
            abs_pos = np.array([0, 60, 0])
            current_node = start_node
            if start_node['children'] is not None: #= if not start_node['name'] = 'end site'
                for child_idx in start_node['children']:
                    child_node = nodes[child_idx]

                    child_offset = np.array(child_node['offset'])
                    qz = pyq.Quaternion(axis=[0, 0, 1], degrees=start_node['rel_degs'][0])
                    qx = pyq.Quaternion(axis=[1, 0, 0], degrees=start_node['rel_degs'][1])
                    qy = pyq.Quaternion(axis=[0, 1, 0], degrees=start_node['rel_degs'][2])
                    qrot = qz * qx * qy
                    offset_rotated = qrot.rotate(child_offset)
                    child_node['rel_pos']= start_node['abs_qt'].rotate(offset_rotated)

                    child_node['abs_qt'] = start_node['abs_qt'] * qrot

            while current_node['parent'] is not None:

                abs_pos = abs_pos + current_node['rel_pos']
                current_node = nodes[current_node['parent']]
            start_node['abs_pos'] = abs_pos

        line = []
        for node in nodes:
            line.append(node['abs_pos'])
        output_lines.append(line)
        
    output_array = np.asarray(output_lines)
    output_vectors = np.empty([len(output_array), n_outputs * 3]) # 3 : x, y, z 
    for idx, line in enumerate(output_array):
        output_vectors[idx] = line.flatten()
    return output_vectors

def get_child_list_from_bvh(bvh_path):
    with open(bvh_path, 'r') as f: 
        bvh_content = f.readlines()
    
    for idx, line in enumerate(bvh_content) : 
        if line.startswith('MOTION'):
            motion_idx = idx 
            break 
    
    hierarchy = bvh_content[: motion_idx]
    nodes = create_hierarchy_nodes(hierarchy)
    return [node['children'] for node in nodes]