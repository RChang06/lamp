import mujoco

URDF = "robot/dummy_lamp_5dof.urdf"


def load_urdf_spec():
    # keep visual meshes and the fixed head, light and camera frames
    src = open(URDF).read()
    src = src.replace(
        '<robot name="dummy_character_lamp_5dof">',
        '<robot name="dummy_character_lamp_5dof"><mujoco><compiler discardvisual="false" '
        'fusestatic="false" meshdir="robot" strippath="false"/></mujoco>',
    ).replace("assets/lamp_shade.stl", "assets/lamp_shade_bin.stl")
    return mujoco.MjSpec.from_string(src)


def build_model():
    spec = load_urdf_spec()

    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    for j in spec.joints:
        effort = j.actfrcrange[1]
        a = spec.add_actuator(name=j.name.replace("_joint", ""), target=j.name,
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=effort / 0.05, kv=effort / 0.5)
        a.ctrlrange = j.range
        a.ctrllimited = True
        j.armature = 0.002

    # the supplied head collision cylinder overlaps the arm at rest
    near = ["upper_arm_link", "neck_link", "head_gimbal_link", "lamp_head_link"]
    for i, a in enumerate(near):
        for b in near[i + 1:]:
            spec.add_exclude(bodyname1=a, bodyname2=b)

    white = spec.add_material(name="white", rgba=[0.95, 0.95, 0.96, 1], specular=0.3, shininess=0.4)
    chrome = spec.add_material(name="chrome", rgba=[0.86, 0.88, 0.93, 1], specular=0.8, shininess=0.8, reflectance=0.05)
    bulb = spec.add_material(name="bulb", rgba=[1.0, 0.93, 0.78, 1], emission=0.6)
    for g in spec.geoms:
        if g.group != 1:
            continue
        r, gr, b = g.rgba[:3]
        if r > 0.99 and b < 0.8:
            g.material = "bulb"
        elif r < 0.1:
            pass
        elif r > 0.9:
            g.material = "white"
        else:
            g.material = "chrome"

    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.42, 0.42, 0.44], rgb2=[0.42, 0.42, 0.44], mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=[0.27, 0.27, 0.29], width=512, height=512)
    floor = spec.add_material(name="floor", texrepeat=[60, 60], reflectance=0.03)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.13, 0.14, 0.16], rgb2=[0.07, 0.07, 0.08], width=512, height=512)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], material="floor", group=2)

    spec.worldbody.add_light(name="key", pos=[1.2, -0.8, 2.2], dir=[-0.45, 0.3, -1], castshadow=True,
                             diffuse=[0.65, 0.62, 0.58], specular=[0.2, 0.2, 0.2], cutoff=60, exponent=2)
    spec.worldbody.add_light(name="fill", pos=[-1.5, 1.0, 1.5], dir=[0.6, -0.4, -0.6], castshadow=False,
                             diffuse=[0.25, 0.26, 0.30], specular=[0, 0, 0])

    head = next(b for b in spec.bodies if b.name == "light_emitter_link")
    head.add_light(name="lamp_light", pos=[0, 0, 0], dir=[1, 0, 0],
                   diffuse=[1.0, 0.85, 0.6], specular=[0.2, 0.2, 0.2], cutoff=45, exponent=8,
                   attenuation=[0.6, 0.4, 0.0], castshadow=False)

    m = spec.compile()
    for g in range(m.ngeom):
        if m.geom_group[g] == 0:
            m.geom_rgba[g, 3] = 0
    m.vis.headlight.ambient[:] = 0.25
    m.vis.headlight.diffuse[:] = 0.35
    m.vis.headlight.specular[:] = 0.0
    m.vis.quality.shadowsize = 4096
    m.vis.quality.offsamples = 8
    m.vis.map.znear = 0.01
    m.vis.global_.offwidth = 1920
    m.vis.global_.offheight = 1080
    return m


REFERENCE_POSE = {"base_yaw": -0.25, "shoulder_pitch": 0.35, "elbow_pitch": -1.2, "neck_yaw": 0.0, "head_pitch": 0.15}
