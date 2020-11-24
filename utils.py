def print_joints(env, p):
    prop_names = ["index", "name", "type", "qIndex", "uIndex", "flags", "damping", "friction", "llimit",
                  "ulimit", "maxforce", "maxvel", "linkname", "jointaxis", "framepos", "frameorn", "pIndex"]
    for i in range(p.getNumJoints(env.robot)):
        jmap = {}
        joint_info = p.getJointInfo(env.robot, i)
        for j in range(len(joint_info)):
            jmap[prop_names[j]] = joint_info[j]
        print(f'Joint {jmap["index"]}')
        print(f'\tName: {jmap["name"]}, Type: {jmap["type"]}, First Position Index: {jmap["qIndex"]}, First Velocity Index: {jmap["uIndex"]}')
        print(f'\tFlags: {jmap["flags"]}, Damping: {jmap["damping"]}, Friction: {jmap["friction"]}, Lower Limit: {jmap["llimit"]}, Upper Limit: {jmap["ulimit"]}')
        print(f'\tMaxForce: {jmap["maxforce"]}, MaxVelocity: {jmap["maxvel"]}, Link Name: {jmap["linkname"]}, Joint Axis: {jmap["jointaxis"]}')
        print(f'\tParent Frame Position: {jmap["framepos"]}')
        print(f'\tParent Frame Orientation: {jmap["frameorn"]}')
        print(f'\tParent Frame Index: {jmap["pIndex"]}')
