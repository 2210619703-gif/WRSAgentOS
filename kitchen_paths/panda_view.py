"""把网格画进 Panda3D 窗口。左键拖动旋转，滚轮缩放，Esc 或关窗口结束。"""

import numpy as np


def capture(body, rgb=None, alpha=None):
    """Copy one posed body's triangles into world coordinates."""
    origin = np.asarray(body.tf, dtype=float)
    batches = []
    for visual in body.visuals:
        geom = visual.geom
        faces = None if geom is None else geom.fs
        if faces is None:
            continue
        local = origin @ np.asarray(visual.loc_tf, dtype=float)
        vs = np.asarray(geom.vs, dtype=float)
        hom = np.concatenate([vs, np.ones((len(vs), 1))], axis=1)
        world = (hom @ local.T)[:, :3]
        color = np.asarray(visual.rgb if rgb is None else rgb, dtype=float)
        batches.append((
            world.astype(np.float32),
            np.asarray(faces, dtype=np.int32),
            color,
            float(visual.alpha if alpha is None else alpha),
        ))
    return batches


def show(batches, title, look, eye, shot=None):
    """Open a Panda3D window. ``shot`` saves one frame and closes, for checks."""
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        AmbientLight,
        CullFaceAttrib,
        DirectionalLight,
        Geom,
        GeomNode,
        GeomTriangles,
        GeomVertexData,
        GeomVertexFormat,
        GeomVertexWriter,
        MouseButton,
        TransparencyAttrib,
        Vec4,
        WindowProperties,
        loadPrcFileData,
    )

    loadPrcFileData("", f"window-title {title}")
    loadPrcFileData("", "win-size 1280 800")
    app = ShowBase()
    app.setBackgroundColor(0.93, 0.94, 0.95, 1)
    props = WindowProperties()
    props.setTitle(title)
    app.win.requestProperties(props)

    root = app.render.attachNewNode("kitchen")
    root.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
    for vertices, faces, rgb, alpha in batches:
        corners = vertices[faces]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(length, 1e-8)
        flat_vs = corners.reshape(-1, 3)
        flat_ns = np.repeat(normals, 3, axis=0)
        data = GeomVertexData("m", GeomVertexFormat.getV3n3(), Geom.UHStatic)
        data.setNumRows(len(flat_vs))
        point = GeomVertexWriter(data, "vertex")
        normal = GeomVertexWriter(data, "normal")
        for vertex, direction in zip(flat_vs, flat_ns):
            point.addData3(float(vertex[0]), float(vertex[1]), float(vertex[2]))
            normal.addData3(float(direction[0]), float(direction[1]), float(direction[2]))
        primitive = GeomTriangles(Geom.UHStatic)
        primitive.addConsecutiveVertices(0, len(flat_vs))
        primitive.closePrimitive()
        geom = Geom(data)
        geom.addPrimitive(primitive)
        node = GeomNode("m")
        node.addGeom(geom)
        path = root.attachNewNode(node)
        path.setColor(float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))
        path.setTransparency(TransparencyAttrib.MAlpha)

    ambient = AmbientLight("ambient")
    ambient.setColor(Vec4(0.55, 0.55, 0.55, 1))
    sun = DirectionalLight("sun")
    sun.setColor(Vec4(0.85, 0.82, 0.75, 1))
    sun_path = app.render.attachNewNode(sun)
    sun_path.setHpr(35, -55, 0)
    app.render.setLight(app.render.attachNewNode(ambient))
    app.render.setLight(sun_path)

    look = np.asarray(look, dtype=float)
    eye = np.asarray(eye, dtype=float)
    offset = eye - look
    state = {
        "look": look,
        "dist": float(np.linalg.norm(offset)),
        "yaw": float(np.degrees(np.arctan2(offset[1], offset[0]))),
        "pitch": float(np.degrees(np.arcsin(np.clip(offset[2] / np.linalg.norm(offset), -1, 1)))),
        "last": None,
    }

    def apply_camera():
        yaw = np.radians(state["yaw"])
        pitch = np.radians(state["pitch"])
        arm = state["dist"] * np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
        ])
        placed = state["look"] + arm
        app.cam.setPos(float(placed[0]), float(placed[1]), float(placed[2]))
        app.cam.lookAt(float(look[0]), float(look[1]), float(look[2]))

    def zoom(scale):
        state["dist"] = max(0.12, state["dist"] * (1.0 + scale))
        apply_camera()

    def orbit(task):
        watcher = app.mouseWatcherNode
        if not watcher.hasMouse() or not watcher.isButtonDown(MouseButton.one()):
            state["last"] = None
            return task.cont
        point = (watcher.getMouseX(), watcher.getMouseY())
        if state["last"] is not None:
            state["yaw"] -= (point[0] - state["last"][0]) * 140
            state["pitch"] = float(np.clip(state["pitch"] + (point[1] - state["last"][1]) * 90, -85, 85))
            apply_camera()
        state["last"] = point
        return task.cont

    app.disableMouse()
    apply_camera()
    app.accept("wheel_up", zoom, [-0.08])
    app.accept("wheel_down", zoom, [0.08])
    app.accept("escape", app.userExit)
    app.taskMgr.add(orbit, "orbit")
    if shot:
        def grab(task):
            if task.frame < 3:
                return task.cont
            app.screenshot(shot, defaultFilename=False)
            app.userExit()
            return task.done

        app.taskMgr.add(grab, "grab")
    print(f"Panda3D 窗口：{title}。左键旋转，滚轮缩放，Esc 关闭。")
    app.run()
