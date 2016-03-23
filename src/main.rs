#[macro_use]
extern crate glium;
extern crate cgmath;

use cgmath::{Vector4, Matrix4};

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 4],
}

fn to_vertex(obj: Vec<Vector4<f32>>) -> Vec<Vertex> {
    let mut result = Vec::new();
    for v in obj {
        result.push(Vertex {position: [v.x, v.y, v.z, v.w]});
    }
    return result;
}

fn transform(t: Matrix4<f32>, obj: Vec<Vector4<f32>>) -> Vec<Vector4<f32>> {
    let mut result = Vec::new();
    for v in obj {
        result.push(t * v);
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

fn scale(x: f32, y: f32, z: f32) -> Matrix4<f32> {
    Matrix4::new(
          x, 0.0, 0.0, 0.0,
        0.0,   y, 0.0, 0.0,
        0.0, 0.0,   z, 0.0,
        0.0, 0.0, 0.0, 1.0
    )
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
            self.stack.push(m * top);
        }
        else {
            self.stack.push(m);
        }
    }
    fn pop(&mut self) {
        self.stack.pop();
    }
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

    implement_vertex!(Vertex, position);

    // body
    let body = vec![
        Vector4::new( 0.5,  0.5,  0.0, 1.0),
        Vector4::new(-0.5,  0.5,  0.0, 1.0),
        Vector4::new( 0.0, -0.5, -0.3, 1.0),
        Vector4::new( 0.0, -0.5,  0.3, 1.0),
    ];

    let body_indices = glium::IndexBuffer::new(
        &display, glium::index::PrimitiveType::TrianglesList,
        &[0u8, 1, 2,
        0, 1, 3,
        0, 2, 3,
        1, 2, 3]
    ).unwrap();

    // arm
    let stick = vec![
        Vector4::new( 0.1,  0.1,  1.0, 1.0),
        Vector4::new( 0.1, -0.1,  1.0, 1.0),
        Vector4::new(-0.1, -0.1,  1.0, 1.0),
        Vector4::new(-0.1,  0.1,  1.0, 1.0),
        Vector4::new( 0.1,  0.1, -1.0, 1.0),
        Vector4::new( 0.1, -0.1, -1.0, 1.0),
        Vector4::new(-0.1, -0.1, -1.0, 1.0),
        Vector4::new(-0.1,  0.1, -1.0, 1.0),
    ];
    let stick_indices = glium::IndexBuffer::new(
        &display, glium::index::PrimitiveType::TrianglesList,
        &[0u8, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
        0, 3, 4, 3, 4, 7,
        6, 7, 2, 7, 2, 3,
        5, 6, 1, 6, 1, 2,
        4, 5, 1, 4, 1, 0]
    ).unwrap();

    let vertex_shader_src = r#"
        #version 330
        in vec4 position;
        uniform vec3 in_color;
        uniform mat4 view;
        uniform mat4 perspective;
        out vec3 _in_color;

        void main() {
            gl_Position = perspective * view * position;
            _in_color = in_color;
        }
    "#;

    let fragment_shader_src = r#"
        #version 330
        in vec3 _in_color;
        out vec4 color;

        void main() {
            color = vec4(_in_color, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();
    let mut t: f32 = 0.0;
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

        let view = view_matrix(&[4.0 * t.cos(), 0.1, 4.0 * t.sin()], &[- 4.0 * t.cos(), -0.1, - 4.0 * t.sin()], &[0.0, 1.0, 0.0]);

        let params = glium::DrawParameters {
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);
        
        matrix_stack.push(translate(0.0, 0.0, 0.0));
        let body_adjust = to_vertex(transform(matrix_stack.top(), body.clone()));
        let body_buffer = glium::VertexBuffer::new(&display, &body_adjust).unwrap();
        target.draw(&body_buffer, &body_indices, &program,
                    &uniform! { view: view, perspective: perspective, in_color: [1.0, 0.0, 0.0f32] },
                    &params).unwrap();

        {
            matrix_stack.push(translate(0.5, 0.5, -0.4));
            let arm_left = to_vertex(transform(matrix_stack.top() * scale(0.7, 0.7, 0.4), stick.clone()));
            let arm_left_buffer = glium::VertexBuffer::new(&display, &arm_left).unwrap();
            target.draw(&arm_left_buffer, &stick_indices, &program,
                        &uniform! { view: view, perspective: perspective, in_color: [1.0, 1.0, 1.0f32] },
                        &params).unwrap();
            matrix_stack.pop();
        }

        {
            matrix_stack.push(translate(-0.5, 0.5, -0.4));
            let arm_right = to_vertex(transform(matrix_stack.top() * scale(0.7, 0.7, 0.4), stick.clone()));
            let arm_right_buffer = glium::VertexBuffer::new(&display, &arm_right).unwrap();
            target.draw(&arm_right_buffer, &stick_indices, &program,
                        &uniform! { view: view, perspective: perspective, in_color: [1.0, 1.0, 1.0f32] },
                        &params).unwrap();
            matrix_stack.pop();
        }

        {
        // body_lower
        matrix_stack.push(translate(0.0, -0.5, 0.0));
        let body_lower = to_vertex(transform(matrix_stack.top() * scale(10.0, 1.0, 0.1), stick.clone()));
        let body_lower_buffer = glium::VertexBuffer::new(&display, &body_lower).unwrap();
        target.draw(&body_lower_buffer, &stick_indices, &program,
                    &uniform! { view: view, perspective: perspective, in_color: [1.0, 1.0, 1.0f32] },
                    &params).unwrap();

        {
        // left leg
        matrix_stack.push(translate(1.0, 0.0, 0.3));
        let leg_upper_left = to_vertex(transform(matrix_stack.top() * scale(1.0, 0.4, 0.3), stick.clone()));
        let leg_upper_left_buffer = glium::VertexBuffer::new(&display, &leg_upper_left).unwrap();
        target.draw(&leg_upper_left_buffer, &stick_indices, &program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 0.0, 1.0f32] },
                    &params).unwrap();

        {
        matrix_stack.push(translate(0.0, -0.3, 0.3));
        let leg_lower_left = to_vertex(transform(matrix_stack.top() * scale(1.0, 3.0, 0.04), stick.clone()));
        let leg_lower_left_buffer = glium::VertexBuffer::new(&display, &leg_lower_left).unwrap();
        target.draw(&leg_lower_left_buffer, &stick_indices, &program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 1.0, 1.0f32] },
                    &params).unwrap();

        matrix_stack.pop();
        }
        matrix_stack.pop();
        }

        {
        // right leg
        matrix_stack.push(translate(-1.0, 0.0, 0.3));
        let leg_upper_right = to_vertex(transform(matrix_stack.top() * scale(1.0, 0.4, 0.3), stick.clone()));
        let leg_upper_right_buffer = glium::VertexBuffer::new(&display, &leg_upper_right).unwrap();
        target.draw(&leg_upper_right_buffer, &stick_indices, &program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 0.0, 1.0f32] },
                    &params).unwrap();

        {
        matrix_stack.push(translate(0.0, -0.3, 0.3));
        let leg_lower_right = to_vertex(transform(matrix_stack.top() * scale(1.0, 3.0, 0.04), stick.clone()));
        let leg_lower_right_buffer = glium::VertexBuffer::new(&display, &leg_lower_right).unwrap();
        target.draw(&leg_lower_right_buffer, &stick_indices, &program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 1.0, 1.0f32] },
                    &params).unwrap();
        matrix_stack.pop();
        }
        matrix_stack.pop();
        }
        matrix_stack.pop();
    }
        matrix_stack.pop();
        target.finish().unwrap();

        for ev in display.poll_events() {
            match ev {
                glium::glutin::Event::Closed => return,
                _ => ()
            }
        }
        t += 0.008;
    }
}
