#[macro_use]
extern crate glium;
extern crate cgmath;
extern crate obj;

use cgmath::*;
use std::fs::File;
use std::io::BufReader;
use obj::*;

#[derive(Copy, Clone)]
struct MyVertex {
    position: [f32; 4],
    normal: [f32; 3],
}

fn obj_to_vertex(obj: Vec<Vertex>) -> Vec<MyVertex> {
    let mut result = Vec::new();
    for v in obj {
        result.push(MyVertex {position: [v.position[0], v.position[1], v.position[2], 1.0], normal: v.normal});
    }
    return result;
}

fn transform(t: Matrix4<f32>, obj: Vec<Vertex>) -> Vec<Vertex> {
    let mut result = Vec::new();
    for v in obj {
        let p = t * Vector4::new(v.position[0], v.position[1], v.position[2], 1.0);
        let n = t * Vector4::new(v.normal[0], v.normal[1], v.normal[2], 1.0);
        result.push(Vertex {position: [p.x, p.y, p.z], normal: [n.x, n.y, n.z]});
    }
    return result;
}

fn translate(x: f32, y: f32, z: f32) -> Matrix4<f32> {
    Matrix4::new(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        x,   y,   z, 1.0
        )
}

fn rotate(axis: &Vector3<f32>, theta: f32) -> Matrix4<f32> {
    let q = Quaternion::new((theta/2.0).cos(), (theta/2.0).sin() * axis.x, (theta/2.0).sin() * axis.y, (theta/2.0).sin() * axis.z);
    let (w, x, y, z) = (q.s, q.v.x, q.v.y, q.v.z);
    Matrix4::new(
        1.0 - 2.0 * y.powi(2) - 2.0 * z.powi(2), 2.0 * x * y - 2.0 * w * z, 2.0 * x * z + 2.0 * w * y, 0.0,
        2.0 * x * y + 2.0 * w * z, 1.0 - 2.0 * x.powi(2) - 2.0 * z.powi(2), 2.0 * y * z - 2.0 * w * x, 0.0,
        2.0 * x * z - 2.0 * w * y, 2.0 * y * z + 2.0 * w * x, 1.0 - 2.0 * x.powi(2) - 2.0 * y.powi(2), 0.0,
        0.0, 0.0, 0.0, 1.0
        ).transpose()
}


#[derive(Clone)]
struct MatrixStack {
    stack: Vec<Matrix4<f32>>,
}

impl MatrixStack {
    fn top(&self) -> Matrix4<f32> {
        match self.stack.get(self.stack.len() - 1) {
            Some(x) => *x,
            None => panic!(),
        }
    }
    fn push(&mut self, m: Matrix4<f32>) {
        if self.stack.len() > 0 {
            let top = self.top();
            self.stack.push(top * m);
        }
        else {
            self.stack.push(m);
        }
    }
    fn pop(&mut self) {
        self.stack.pop();
    }
}

fn to_uniform(m: Matrix4<f32>) -> [[f32; 4]; 4] {
    [
        [m.x.x, m.x.y, m.x.z, m.x.w],
        [m.y.x, m.y.y, m.y.z, m.y.w],
        [m.z.x, m.z.y, m.z.z, m.z.w],
        [m.w.x, m.w.y, m.w.z, m.w.w],
    ]
}

// Cited from https://trgk.gitbooks.io/glium-tutorial-russian/content/tuto-12-camera.html
fn view_matrix(position: &[f32; 3], direction: &[f32; 3], up: &[f32; 3]) -> [[f32; 4]; 4] {
    let f = {
        let f = direction;
        let len = f[0] * f[0] + f[1] * f[1] + f[2] * f[2];
        let len = len.sqrt();
        [f[0] / len, f[1] / len, f[2] / len]
    };

    let s = [up[1] * f[2] - up[2] * f[1],
    up[2] * f[0] - up[0] * f[2],
    up[0] * f[1] - up[1] * f[0]];

    let s_norm = {
        let len = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
        let len = len.sqrt();
        [s[0] / len, s[1] / len, s[2] / len]
    };

    let u = [f[1] * s_norm[2] - f[2] * s_norm[1],
    f[2] * s_norm[0] - f[0] * s_norm[2],
    f[0] * s_norm[1] - f[1] * s_norm[0]];

    let p = [-position[0] * s_norm[0] - position[1] * s_norm[1] - position[2] * s_norm[2],
    -position[0] * u[0] - position[1] * u[1] - position[2] * u[2],
    -position[0] * f[0] - position[1] * f[1] - position[2] * f[2]];

    [
        [s[0], u[0], f[0], 0.0],
        [s[1], u[1], f[1], 0.0],
        [s[2], u[2], f[2], 0.0],
        [p[0], p[1], p[2], 1.0],
        ]
}

fn main() {
    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new().build_glium().unwrap();

    implement_vertex!(MyVertex, position, normal);

    let body: Obj = load_obj(BufReader::new(File::open("./models/body.obj").unwrap())).unwrap();
    let body_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &body.indices).unwrap();
    let body_lower: Obj = load_obj(BufReader::new(File::open("./models/body_lower.obj").unwrap())).unwrap();
    let body_lower_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &body_lower.indices).unwrap();
    let ankle: Obj = load_obj(BufReader::new(File::open("./models/ankle.obj").unwrap())).unwrap();
    let ankle_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &ankle.indices).unwrap();
    let gun: Obj = load_obj(BufReader::new(File::open("./models/gun.obj").unwrap())).unwrap();
    let gun_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &gun.indices).unwrap();
    let foot: Obj = load_obj(BufReader::new(File::open("./models/foot.obj").unwrap())).unwrap();
    let foot_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &foot.indices).unwrap();

    let arm_left: Obj = load_obj(BufReader::new(File::open("./models/arm_left.obj").unwrap())).unwrap();
    let arm_left_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &arm_left.indices).unwrap();
    let leg_upper_left: Obj = load_obj(BufReader::new(File::open("./models/leg_upper_left.obj").unwrap())).unwrap();
    let leg_upper_left_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &leg_upper_left.indices).unwrap();
    let leg_lower_left: Obj = load_obj(BufReader::new(File::open("./models/leg_lower_left.obj").unwrap())).unwrap();
    let leg_lower_left_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &leg_lower_left.indices).unwrap();

    let arm_right: Obj = load_obj(BufReader::new(File::open("./models/arm_right.obj").unwrap())).unwrap();
    let arm_right_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &arm_right.indices).unwrap();
    let leg_upper_right: Obj = load_obj(BufReader::new(File::open("./models/leg_upper_right.obj").unwrap())).unwrap();
    let leg_upper_right_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &leg_upper_right.indices).unwrap();
    let leg_lower_right: Obj = load_obj(BufReader::new(File::open("./models/leg_lower_right.obj").unwrap())).unwrap();
    let leg_lower_right_indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList, &leg_lower_right.indices).unwrap();

    let vertex_shader_src = r#"
        #version 330
        in vec4 position;
        in vec3 normal;

        out vec3 v_normal;

        uniform vec3 in_color;
        uniform mat4 transform;
        uniform mat4 view;
        uniform mat4 perspective;
        out vec3 _in_color;

        void main() {
            gl_Position = perspective * view * transform * position;
            v_normal = (transform * vec4(normal, 0.0)).xyz;
            _in_color = in_color;
        }
    "#;

    let fragment_shader_src = r#"
        #version 330
        in vec3 _in_color;
        in vec3 v_normal;
        out vec4 color;
        uniform vec3 u_light;

        void main() {
            float brightness = dot(normalize(v_normal), normalize(u_light));
            vec3 dark_color = 0.6 * _in_color;
            color = vec4(mix(dark_color, _in_color, brightness), 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();
    let mut t: f32 = 0.0;
    let mut cam_angle: f32 = 0.0;
    let mut arm_angle: f32 = 0.0;
    let mut body_angle: f32 = 0.0;
    let mut matrix_stack = MatrixStack {stack: Vec::new()};

    loop {
        let mut target = display.draw();
        let perspective = {
            let (width, height) = target.get_dimensions();
            let aspect_ratio = height as f32 / width as f32;

            let fov: f32 = 3.141592 / 3.0;
            let zfar = 1024.0;
            let znear = 0.1;

            let f = 1.0 / (fov / 2.0).tan();

            [
                [f *   aspect_ratio   ,    0.0,              0.0              ,   0.0],
                [         0.0         ,     f ,              0.0              ,   0.0],
                [         0.0         ,    0.0,  (zfar+znear)/(zfar-znear)    ,   1.0],
                [         0.0         ,    0.0, -(2.0*zfar*znear)/(zfar-znear),   0.0],
                ]
        };

        let view = view_matrix(&[15.0 * cam_angle.cos() + 1.98, 10.0, 15.0 * cam_angle.sin()], &[- 15.0 * cam_angle.cos(), -7.0, - 15.0 * cam_angle.sin()], &[0.0, 1.0, 0.0]);
        // let view = view_matrix(&[15.0 + 1.98, 10.0, 0.0], &[- 15.0, -7.0, 0.0], &[0.0, 1.0, 0.0]);

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        target.clear_color_and_depth((0.5, 0.5, 0.5, 1.0), 1.0);

        let light = [1.0, 0.0, 0.0f32];

        // body_lower
        matrix_stack.push(translate(2.01, 5.33, 1.13));
        let body_lower_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(body_lower.vertices.clone())).unwrap();
        target.draw(&body_lower_buffer, &body_lower_indices, &program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 0.2, 0.2f32], u_light: light,
                    transform: to_uniform(matrix_stack.top())},
                    &params).unwrap();
        {
            matrix_stack.push(translate(0.1, 1.27, -0.6) * rotate(&Vector3::new(0.0, 1.0, 0.0), body_angle));
            let body_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(body.vertices.clone())).unwrap();
            target.draw(&body_buffer, &body_indices, &program,
                        &uniform! { view: view, perspective: perspective, in_color: [0.0, 0.0, 0.2f32], u_light: light,
                        transform: to_uniform(matrix_stack.top())},
                        &params).unwrap();

            {
                matrix_stack.push(translate(-2.72, 0.85, -0.73) * rotate(&Vector3::new(-1.0, 0.0, 0.0), arm_angle));
                let arm_left_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(arm_left.vertices.clone())).unwrap();
                target.draw(&arm_left_buffer, &arm_left_indices, &program,
                            &uniform! { view: view, perspective: perspective, in_color: [1.0, 0.03, 0.0f32], u_light: light,
                            transform: to_uniform(matrix_stack.top())},
                            &params).unwrap();
                {
                    matrix_stack.push(translate(0.37, 0.16, -3.54));
                    let gun_left_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(transform(rotate(&Vector3::new(0.0, 0.0, 1.0), t * 7.0), gun.vertices.clone()))).unwrap();
                    target.draw(&gun_left_buffer, &gun_indices, &program,
                                &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.0, 0.0f32], u_light: light,
                                transform: to_uniform(matrix_stack.top())},
                                &params).unwrap();
                    matrix_stack.pop();
                }
                matrix_stack.pop();
            }
            {
                matrix_stack.push(translate(2.72, 0.85, -0.73) * rotate(&Vector3::new(-1.0, 0.0, 0.0), arm_angle));
                let arm_right_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(arm_right.vertices.clone())).unwrap();
                target.draw(&arm_right_buffer, &arm_right_indices, &program,
                            &uniform! { view: view, perspective: perspective, in_color: [1.0, 0.03, 0.0f32], u_light: light,
                            transform: to_uniform(matrix_stack.top())},
                            &params).unwrap();
                {
                    matrix_stack.push(translate(-0.37, 0.16, -3.54));
                    let gun_right_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(transform(rotate(&Vector3::new(0.0, 0.0, 1.0), t * -7.0), gun.vertices.clone()))).unwrap();
                    target.draw(&gun_right_buffer, &gun_indices, &program,
                                &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.0, 0.0f32], u_light: light,
                                transform: to_uniform(matrix_stack.top())},
                                &params).unwrap();
                    matrix_stack.pop();
                }
                matrix_stack.pop();
            }
            matrix_stack.pop();
        }

        {
            // left upper
            matrix_stack.push(translate(-2.02, -1.36, 1.33));
            let leg_upper_left_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(leg_upper_left.vertices.clone())).unwrap();
            target.draw(&leg_upper_left_buffer, &leg_upper_left_indices, &program,
                        &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.1, 0.08f32], u_light: light,
                        transform: to_uniform(matrix_stack.top())},
                        &params).unwrap();

            {
                // left ankle
                matrix_stack.push(translate(0.0, -1.43, 1.37));
                let ankle_left_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(ankle.vertices.clone())).unwrap();
                target.draw(&ankle_left_buffer, &ankle_indices, &program,
                            &uniform! { view: view, perspective: perspective, in_color: [0.1, 0.05, 0.04f32], u_light: light,
                            transform: to_uniform(matrix_stack.top())},
                            &params).unwrap();

                {
                    // left lower
                    matrix_stack.push(translate(0.0, -1.37, -0.76));
                    let leg_lower_left_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(leg_lower_left.vertices.clone())).unwrap();
                    target.draw(&leg_lower_left_buffer, &leg_lower_left_indices, &program,
                                &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.0, 0.0f32], u_light: light,
                                transform: to_uniform(matrix_stack.top())},
                                &params).unwrap();

                    {
                        // left foot
                        matrix_stack.push(translate(0.0, -1.17, -0.81));
                        let foot_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(foot.vertices.clone())).unwrap();
                        target.draw(&foot_buffer, &foot_indices, &program,
                                    &uniform! { view: view, perspective: perspective, in_color: [0.1, 0.1, 0.1f32], u_light: light,
                                    transform: to_uniform(matrix_stack.top())},
                                    &params).unwrap();

                        matrix_stack.pop();
                    }
                    matrix_stack.pop();
                }
                matrix_stack.pop();
            }
            matrix_stack.pop();
        }
        {
            // right upper
            matrix_stack.push(translate(2.02, -1.36, 1.33));
            let leg_upper_right_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(leg_upper_right.vertices.clone())).unwrap();
            target.draw(&leg_upper_right_buffer, &leg_upper_right_indices, &program,
                        &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.1, 0.08f32], u_light: light,
                        transform: to_uniform(matrix_stack.top())},
                        &params).unwrap();

            {
                // right ankle
                matrix_stack.push(translate(0.0, -1.43, 1.37));
                let ankle_right_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(ankle.vertices.clone())).unwrap();
                target.draw(&ankle_right_buffer, &ankle_indices, &program,
                            &uniform! { view: view, perspective: perspective, in_color: [0.1, 0.05, 0.04f32], u_light: light,
                            transform: to_uniform(matrix_stack.top())},
                            &params).unwrap();

                {
                    // right lower
                    matrix_stack.push(translate(0.0, -1.37, -0.76));
                    let leg_lower_right_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(leg_lower_right.vertices.clone())).unwrap();
                    target.draw(&leg_lower_right_buffer, &leg_lower_right_indices, &program,
                                &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.0, 0.0f32], u_light: light,
                                transform: to_uniform(matrix_stack.top())},
                                &params).unwrap();

                    {
                        // right foot
                        matrix_stack.push(translate(0.0, -1.17, -0.81));
                        let foot_buffer = glium::VertexBuffer::new(&display, &obj_to_vertex(foot.vertices.clone())).unwrap();
                        target.draw(&foot_buffer, &foot_indices, &program,
                                    &uniform! { view: view, perspective: perspective, in_color: [0.1, 0.1, 0.1f32], u_light: light,
                                    transform: to_uniform(matrix_stack.top())},
                                    &params).unwrap();

                        matrix_stack.pop();
                    }
                    matrix_stack.pop();
                }
                matrix_stack.pop();
            }
            matrix_stack.pop();
        }
        matrix_stack.pop();
        target.finish().unwrap();

        for ev in display.poll_events() {
            use glium::glutin::*;
            match ev {
                Event::Closed => return,
                Event::KeyboardInput(ElementState::Pressed, _, Some(key)) => {
                    match key {
                        VirtualKeyCode::Left  => { body_angle -= 0.04; },
                        VirtualKeyCode::Right => { body_angle += 0.04; },
                        VirtualKeyCode::Up  =>   { arm_angle -= 0.04; },
                        VirtualKeyCode::Down =>  { arm_angle += 0.04; },
                        VirtualKeyCode::A   =>   { cam_angle -= 0.04; },
                        VirtualKeyCode::D   =>   { cam_angle += 0.04; },
                        _ => ()
                    }
                },
                _ => ()
            }
        }
        t += 0.008;
    }
}
