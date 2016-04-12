#[macro_use]
extern crate glium;
extern crate gmath;

use gmath::matrix::matrix4::*;
use gmath::vector::vector3::*;
use gmath::vector::vector4::*;
use gmath::view::*;
use gmath::model::*;
use gmath::quaternion::*;

#[derive(Clone)]
struct MatrixStack {
    stack: Vec<Matrix4>,
}

impl MatrixStack {
    fn top(&self) -> Matrix4 {
        match self.stack.get(self.stack.len() - 1) {
            Some(x) => *x,
            None => panic!(),
        }
    }
    fn push(&mut self, m: Matrix4) {
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

enum MouseState {
    Released,
    LClicked(i32, i32),
    RClicked(i32, i32),
}

fn circle(center: Vector3, axis: Vector3, r: f32) -> Vec<ModelVertex> {
    let n = 30;
    let pi = 3.141592;
    let mut result = Vec::new();
    let z = Vector3::new(0.0, 0.0, 1.0);
    let q = if axis == z {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }
    else {
        Quaternion::rotate_angle_axis(Vector3::dot(z, axis.normalize()).acos(), Vector3::cross(z, axis))
    };
    for i in 0..n {
        let theta: f32 = (2.0 * pi * i as f32) / n as f32;
        let position = Matrix4::translate(center.x, center.y, center.z) * q.to_matrix() * Vector4::new(r * theta.cos(), r * theta.sin(), 0.0, 1.0);
        let normal =   q.to_matrix() * Vector4::new(theta.cos(), theta.sin(), 0.0, 0.0);
        result.push(ModelVertex {
            position: Vector3::new(position.x, position.y, position.z),
            normal:   Vector3::new(normal.x, normal.y, normal.z)
        });
    }
    return result;
}

fn main() {
    use glium::{DisplayBuild, Surface};
    let display = glium::glutin::WindowBuilder::new().build_glium().unwrap();

    let (body_lower_buffer, body_lower_indices) = load_object(&display, "./models/body_lower.obj");
    let (body_buffer, body_indices) = load_object(&display, "./models/body.obj");
    let (ankle_buffer, ankle_indices) = load_object(&display, "./models/ankle.obj");
    let (gun_buffer, gun_indices) = load_object(&display, "./models/gun.obj");
    let (foot_buffer, foot_indices) = load_object(&display, "./models/foot.obj");

    let (arm_left_buffer, arm_left_indices) = load_object(&display, "./models/arm_left.obj");
    let (leg_upper_left_buffer, leg_upper_left_indices) = load_object(&display, "./models/leg_upper_left.obj");
    let (leg_lower_left_buffer, leg_lower_left_indices) = load_object(&display, "./models/leg_lower_left.obj");

    let (arm_right_buffer, arm_right_indices) = load_object(&display, "./models/arm_right.obj");
    let (leg_upper_right_buffer, leg_upper_right_indices) = load_object(&display, "./models/leg_upper_right.obj");
    let (leg_lower_right_buffer, leg_lower_right_indices) = load_object(&display, "./models/leg_lower_right.obj");

    let vertex_shader_src = r#"
        #version 130
        in vec3 position;
        in vec3 normal;

        out vec3 v_normal;

        uniform vec3 in_color;
        uniform mat4 transform;
        uniform mat4 view;
        uniform mat4 perspective;
        out vec3 _in_color;

        void main() {
            gl_Position = perspective * view * transform * vec4(position, 1.0);
            v_normal = (transform * vec4(normal, 0.0)).xyz;
            _in_color = in_color;
        }
    "#;

    let fragment_shader_src = r#"
        #version 130
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

    let ui_fragment_shader_src = r#"
        #version 130
        in vec3 _in_color;
        in vec3 v_normal;
        out vec4 color;
        uniform vec3 u_light;

        void main() {
            color = vec4(_in_color, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();
    let ui_program = glium::Program::from_source(&display, vertex_shader_src, ui_fragment_shader_src, None).unwrap();
    let mut t: f32 = 0.0;
    let arm_angle = 0.0;
    let body_angle = 0.0;
    let mut matrix_stack = MatrixStack {stack: Vec::new()};

    let mut state = MouseState::Released;
    let mut x: i32 = 0;
    let mut y: i32 = 0;
    let mut fov = 3.141592 / 3.0;

    let center: Vector3 = Vector3::zero();
    let mut cam_pos = Vector3::new(0.0, 0.0, -15.0);
    let mut view_center = center;
    let mut view_up = Vector3::new(0.0, 1.0, 0.0);
    let mut shift: bool = false;


    loop {
        let mut target = display.draw();
        let (width, height) = target.get_dimensions();
        let perspective = gmath::view::perspective(width, height, fov);
        let view = gmath::view::view_matrix(&cam_pos, &(view_center - cam_pos).normalize(),
                                            &view_up);

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
        let x_circle = glium::VertexBuffer::new(&display, &circle(view_center, Vector3::new(1.0, 0.0, 0.0), 7.0)).unwrap();
        let y_circle = glium::VertexBuffer::new(&display, &circle(view_center, Vector3::new(0.0, 1.0, 0.0), 7.0)).unwrap();
        let z_circle = glium::VertexBuffer::new(&display, &circle(view_center, Vector3::new(0.0, 0.0, 1.0), 7.0)).unwrap();
        let x_line = glium::VertexBuffer::new(&display, &vec![
                                              ModelVertex { position: Vector3::new(-2048.0, 0.0, 0.0), normal: Vector3::zero() },
                                              ModelVertex { position: Vector3::new( 2048.0, 0.0, 0.0), normal: Vector3::zero() }
                                              ]).unwrap();
        let y_line = glium::VertexBuffer::new(&display, &vec![
                                              ModelVertex { position: Vector3::new(0.0, -2048.0, 0.0), normal: Vector3::zero() },
                                              ModelVertex { position: Vector3::new(0.0,  2048.0, 0.0), normal: Vector3::zero() }
                                              ]).unwrap();
        let z_line = glium::VertexBuffer::new(&display, &vec![
                                              ModelVertex { position: Vector3::new(0.0, 0.0, -2048.0), normal: Vector3::zero() },
                                              ModelVertex { position: Vector3::new(0.0, 0.0,  2048.0), normal: Vector3::zero() }
                                              ]).unwrap();
        let circle_indices = glium::index::NoIndices(glium::index::PrimitiveType::LineLoop);
        let line_indices = glium::index::NoIndices(glium::index::PrimitiveType::LinesList);

        target.draw(&x_circle, &circle_indices, &ui_program,
                    &uniform! { view: view, perspective: perspective, in_color: [1.0, 0.0, 0.0f32], u_light: light,
                    transform: Matrix4::one()},
                    &params).unwrap();
        target.draw(&y_circle, &circle_indices, &ui_program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 1.0, 0.0f32], u_light: light,
                    transform: Matrix4::one()},
                    &params).unwrap();
        target.draw(&z_circle, &circle_indices, &ui_program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 0.0, 1.0f32], u_light: light,
                    transform: Matrix4::one()},
                    &params).unwrap();
        target.draw(&x_line, &line_indices, &ui_program,
                    &uniform! { view: view, perspective: perspective, in_color: [1.0, 0.0, 0.0f32], u_light: light,
                    transform: Matrix4::one()},
                    &params).unwrap();
        target.draw(&y_line, &line_indices, &ui_program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 1.0, 0.0f32], u_light: light,
                    transform: Matrix4::one()},
                    &params).unwrap();
        target.draw(&z_line, &line_indices, &ui_program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 0.0, 1.0f32], u_light: light,
                    transform: Matrix4::one()},
                    &params).unwrap();

        matrix_stack.push(Matrix4::translate(0.0, 0.0, 0.0));
        target.draw(&body_lower_buffer, &body_lower_indices, &program,
                    &uniform! { view: view, perspective: perspective, in_color: [0.0, 0.2, 0.2f32], u_light: light,
                    transform: matrix_stack.top()},
                    &params).unwrap();
        {
            matrix_stack.push(Matrix4::translate(0.1, 1.27, -0.6) * Quaternion::rotate_angle_axis(body_angle, Vector3::new(0.0, 1.0, 0.0)).to_matrix());
            target.draw(&body_buffer, &body_indices, &program,
                        &uniform! { view: view, perspective: perspective, in_color: [0.0, 0.0, 0.2f32], u_light: light,
                        transform: matrix_stack.top()},
                        &params).unwrap();

            {
                matrix_stack.push(Matrix4::translate(-2.72, 0.85, -0.73) * Quaternion::rotate_angle_axis(arm_angle, Vector3::new(-1.0, 0.0, 0.0)).to_matrix());
                target.draw(&arm_left_buffer, &arm_left_indices, &program,
                            &uniform! { view: view, perspective: perspective, in_color: [1.0, 0.03, 0.0f32], u_light: light,
                            transform: matrix_stack.top()},
                            &params).unwrap();
                {
                    matrix_stack.push(Matrix4::translate(0.37, 0.16, -3.54) * Quaternion::rotate_angle_axis(t * 7.0, Vector3::new(0.0, 0.0, 1.0)).to_matrix());
                    target.draw(&gun_buffer, &gun_indices, &program,
                                &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.0, 0.0f32], u_light: light,
                                transform: matrix_stack.top()},
                                &params).unwrap();
                    matrix_stack.pop();
                }
                matrix_stack.pop();
            }
            {
                matrix_stack.push(Matrix4::translate(2.72, 0.85, -0.73) * Quaternion::rotate_angle_axis(arm_angle, Vector3::new(-1.0, 0.0, 0.0)).to_matrix());
                target.draw(&arm_right_buffer, &arm_right_indices, &program,
                            &uniform! { view: view, perspective: perspective, in_color: [1.0, 0.03, 0.0f32], u_light: light,
                            transform: matrix_stack.top()},
                            &params).unwrap();
                {
                    matrix_stack.push(Matrix4::translate(-0.37, 0.16, -3.54) * Quaternion::rotate_angle_axis(t * 7.0, Vector3::new(0.0, 0.0, 1.0)).to_matrix());
                    target.draw(&gun_buffer, &gun_indices, &program,
                                &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.0, 0.0f32], u_light: light,
                                transform: matrix_stack.top()},
                                &params).unwrap();
                    matrix_stack.pop();
                }
                matrix_stack.pop();
            }
            matrix_stack.pop();
        }

        {
            // left upper
            matrix_stack.push(Matrix4::translate(-2.02, -1.36, 1.33));
            target.draw(&leg_upper_left_buffer, &leg_upper_left_indices, &program,
                        &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.1, 0.08f32], u_light: light,
                        transform: matrix_stack.top()},
                        &params).unwrap();

            {
                // left ankle
                matrix_stack.push(Matrix4::translate(0.0, -1.43, 1.37));
                target.draw(&ankle_buffer, &ankle_indices, &program,
                            &uniform! { view: view, perspective: perspective, in_color: [0.1, 0.05, 0.04f32], u_light: light,
                            transform: matrix_stack.top()},
                            &params).unwrap();

                {
                    // left lower
                    matrix_stack.push(Matrix4::translate(0.0, -1.37, -0.76));
                    target.draw(&leg_lower_left_buffer, &leg_lower_left_indices, &program,
                                &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.0, 0.0f32], u_light: light,
                                transform: matrix_stack.top()},
                                &params).unwrap();

                    {
                        // left foot
                        matrix_stack.push(Matrix4::translate(0.0, -1.17, -0.81));
                        target.draw(&foot_buffer, &foot_indices, &program,
                                    &uniform! { view: view, perspective: perspective, in_color: [0.1, 0.1, 0.1f32], u_light: light,
                                    transform: matrix_stack.top()},
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
            matrix_stack.push(Matrix4::translate(2.02, -1.36, 1.33));
            target.draw(&leg_upper_right_buffer, &leg_upper_right_indices, &program,
                        &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.1, 0.08f32], u_light: light,
                        transform: matrix_stack.top()},
                        &params).unwrap();

            {
                // right ankle
                matrix_stack.push(Matrix4::translate(0.0, -1.43, 1.37));
                target.draw(&ankle_buffer, &ankle_indices, &program,
                            &uniform! { view: view, perspective: perspective, in_color: [0.1, 0.05, 0.04f32], u_light: light,
                            transform: matrix_stack.top()},
                            &params).unwrap();

                {
                    // right lower
                    matrix_stack.push(Matrix4::translate(0.0, -1.37, -0.76));
                    target.draw(&leg_lower_right_buffer, &leg_lower_right_indices, &program,
                                &uniform! { view: view, perspective: perspective, in_color: [0.2, 0.0, 0.0f32], u_light: light,
                                transform: matrix_stack.top()},
                                &params).unwrap();

                    {
                        // right foot
                        matrix_stack.push(Matrix4::translate(0.0, -1.17, -0.81));
                        target.draw(&foot_buffer, &foot_indices, &program,
                                    &uniform! { view: view, perspective: perspective, in_color: [0.1, 0.1, 0.1f32], u_light: light,
                                    transform: matrix_stack.top()},
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
                        VirtualKeyCode::Up  => {
                            if (cam_pos - view_center).abs() > 5.0 {
                                cam_pos = cam_pos + 0.5 * (view_center - cam_pos).normalize();
                            }
                        },
                        VirtualKeyCode::Down => {
                            cam_pos = cam_pos - 0.5 * (view_center - cam_pos).normalize();
                        },
                        VirtualKeyCode::PageUp  =>   {
                            if fov > 0.1 {
                                fov -= 0.1;
                            }
                        },
                        VirtualKeyCode::PageDown =>  {
                            if fov < 3.141592 / 2.0 - 0.1 {
                                fov += 0.1;
                            }
                        },
                        VirtualKeyCode::Space => {
                            fov = 3.141592 / 3.0;
                            view_center = center;
                            // cam_pos = - 15.0 * (view_center - cam_pos).normalize();
                            view_up = Vector3::new(0.0, 1.0, 0.0);
                            cam_pos = Vector3::new(0.0, 0.0, -15.0);
                        },
                        VirtualKeyCode::LShift | VirtualKeyCode::RShift => {
                            shift = true;
                        }
                        _ => ()
                    }
                },
                Event::KeyboardInput(ElementState::Released, _, Some(key)) => {
                    match key {
                        VirtualKeyCode::LShift | VirtualKeyCode::RShift => {
                            shift = false;
                        },
                        _ => (),
                    }
                },
                Event::MouseMoved((x_pos, y_pos)) => {
                    x = x_pos;
                    y = y_pos;
                },
                Event::MouseInput(ElementState::Pressed, MouseButton::Left) => {
                    if shift {
                        view_center = screen2zplane(perspective, view, x, y, width, height, view_center);
                    }
                    else {
                        state = MouseState::LClicked(x, y);
                    }
                },
                Event::MouseInput(ElementState::Pressed, MouseButton::Right) => {
                    state = MouseState::RClicked(x, y);
                },
                Event::MouseInput(ElementState::Released, _) => {
                    state = MouseState::Released;
                },
                _ => ()
            }
        }

        match state {
            MouseState::Released => (),
            MouseState::LClicked(x_clk, y_clk) => {
                let v1 = screen2ball(perspective, view, x_clk, y_clk, width, height, cam_pos, view_center, 7.0);
                let v2 = screen2ball(perspective, view, x, y, width, height, cam_pos, view_center, 7.0);
                if v1 != v2 {
                    let v1 = v1 - view_center;
                    let v2 = v2 - view_center;
                    let q = Quaternion::rotate_angle_axis(-Vector3::dot(v1.normalize(), v2.normalize()).acos(), Vector3::cross(v1, v2)).to_matrix();
                    cam_pos = (Matrix4::translate(view_center.x, view_center.y, view_center.z) * q
                               * Matrix4::translate(-view_center.x, -view_center.y, -view_center.z) * cam_pos.extend())
                        .truncate();
                    view_up = (q * view_up.extend()).truncate();
                }
                state = MouseState::LClicked(x, y);
            },
            MouseState::RClicked(x_clk, y_clk) => {
                let v1 = screen2zplane(perspective, view, x_clk, y_clk, width, height, view_center);
                let v2 = screen2zplane(perspective, view, x, y, width, height, view_center);
                cam_pos = cam_pos + (v1 - v2);
                view_center = view_center + (v1 - v2);
                state = MouseState::RClicked(x, y);
            },
        }

        t += 0.008;
    }
}

fn screen2ball(perspective: Matrix4, view: Matrix4, x: i32, y: i32, width: u32, height: u32, cam_pos: Vector3, center: Vector3, r: f32) -> Vector3 {
    let pivot = perspective * view * center.extend();
    let pivot = pivot / pivot.w;
    let xp = (x as f32 / width as f32) * 2.0 - 1.0;
    let yp = 1.0 - (y as f32 / height as f32) * 2.0;
    let sp = view.inverse() * perspective.inverse() * Vector4::new(xp, yp, pivot.z, 1.0);
    let sp = sp / sp.w;
    let v = (sp.truncate() - cam_pos).normalize();
    let h = cam_pos + Vector3::dot((center - cam_pos), v) * v;
    let len = (r as f32).powi(2) - (h - center).abs().powi(2);
    if len > 0.0 {
        return h - v * len.sqrt();
    }
    else {
        return center + (h - center).normalize() * r;
        // return screen2zplane(perspective, view, x, y, width, height, center);
    }
}

fn screen2zplane(perspective: Matrix4, view: Matrix4, x: i32, y: i32, width: u32, height: u32, center: Vector3) -> Vector3 {
    let pivot = perspective * view * center.extend();
    let pivot = pivot / pivot.w;
    let xp = (x as f32 / width as f32) * 2.0 - 1.0;
    let yp = 1.0 - (y as f32 / height as f32) * 2.0;
    let sp = view.inverse() * perspective.inverse() * Vector4::new(xp, yp, pivot.z, 1.0);
    let sp = sp / sp.w;
    return sp.truncate();
}
