#[macro_use]
extern crate glium;
extern crate gmath;

use gmath::matrix::matrix4::*;
use gmath::vector::vector3::*;
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
        #version 330
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
    let d: f32 = 15.0;
    let mut cam_angle: f32 = 0.0;
    let mut arm_angle: f32 = 0.0;
    let mut body_angle: f32 = 0.0;
    let mut matrix_stack = MatrixStack {stack: Vec::new()};

    loop {
        let mut target = display.draw();
        let (width, height) = target.get_dimensions();
        let perspective = gmath::view::perspective(width, height, 3.141592 / 3.0).transpose();
        let view = gmath::view::view_matrix(&Vector3::new(d * cam_angle.cos() + 1.98, 10.0, d * cam_angle.sin()),
                                            &Vector3::new(- d * cam_angle.cos(), -7.0, - d * cam_angle.sin()),
                                            &Vector3::new(0.0, 1.0, 0.0)).transpose();

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
        matrix_stack.push(Matrix4::translate(2.01, 5.33, 1.13));
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
