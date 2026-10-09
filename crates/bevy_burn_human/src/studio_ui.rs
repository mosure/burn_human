//! Retained studio controls using Bevy's own UI and Feathers widgets.
//! Forms rebuild only when their structure changes; values update in place.
use bevy::{
    feathers::{
        FeathersPlugins, controls::*, dark_theme::create_dark_theme, display::caption,
        theme::UiTheme,
    },
    input::mouse::{MouseScrollUnit, MouseWheel},
    input_focus::{
        InputFocus,
        tab_navigation::{TabGroup, TabIndex},
    },
    prelude::*,
    text::{EditableText, FontSize, TextCursorStyle, TextEdit, TextEditChange},
    ui::{Checked, InteractionDisabled},
    ui_widgets::{Activate, SliderPrecision, SliderStep, SliderValue, TextInput, ValueChange},
    window::PrimaryWindow,
};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum Value {
    Action,
    Text(String),
    Number(f32),
    Toggle(bool),
    Choice(usize),
    Point(Vec2),
}
#[derive(Clone, Debug, Message)]
pub struct ControlEvent {
    pub id: String,
    pub value: Value,
}
#[derive(Clone, Debug)]
pub enum Row {
    Label(String),
    Buttons(Vec<(String, String, bool)>),
    Button {
        id: String,
        label: String,
        enabled: bool,
    },
    Text {
        id: String,
        label: String,
        value: String,
        multiline: bool,
    },
    Slider {
        id: String,
        label: String,
        value: f32,
        min: f32,
        max: f32,
        step: f32,
    },
    Toggle {
        id: String,
        label: String,
        value: bool,
    },
    Select {
        id: String,
        label: String,
        options: Vec<String>,
        value: usize,
    },
    Image {
        id: String,
        image: Handle<Image>,
        aspect: f32,
        crop: [f32; 4],
        points: Vec<[f32; 2]>,
    },
}
impl Row {
    pub fn button(id: &str, label: &str, enabled: bool) -> Self {
        Self::Button {
            id: id.into(),
            label: label.into(),
            enabled,
        }
    }
    pub fn text(id: &str, label: &str, value: &str) -> Self {
        Self::Text {
            id: id.into(),
            label: label.into(),
            value: value.into(),
            multiline: false,
        }
    }
    pub fn slider(id: &str, label: &str, value: f32, min: f32, max: f32, step: f32) -> Self {
        Self::Slider {
            id: id.into(),
            label: label.into(),
            value,
            min,
            max,
            step,
        }
    }
    pub fn toggle(id: &str, label: &str, value: bool) -> Self {
        Self::Toggle {
            id: id.into(),
            label: label.into(),
            value,
        }
    }
    pub fn select(id: &str, label: &str, options: Vec<String>, value: usize) -> Self {
        Self::Select {
            id: id.into(),
            label: label.into(),
            options,
            value,
        }
    }
    fn structure(&self) -> String {
        match self {
            Self::Label(_) => "label".into(),
            Self::Buttons(items) => format!(
                "buttons:{:?}",
                items
                    .iter()
                    .map(|(id, label, _)| (id, label))
                    .collect::<Vec<_>>()
            ),
            Self::Button { id, label, .. } => format!("button:{id}:{label}"),
            Self::Text {
                id,
                label,
                multiline,
                ..
            } => format!("text:{id}:{label}:{multiline}"),
            Self::Slider {
                id,
                label,
                min,
                max,
                step,
                ..
            } => format!("slider:{id}:{label}:{min}:{max}:{step}"),
            Self::Toggle { id, label, .. } => format!("toggle:{id}:{label}"),
            Self::Select {
                id,
                label,
                options,
                value,
            } => format!("select:{id}:{label}:{options:?}:{value}"),
            Self::Image {
                id,
                image,
                aspect,
                points,
                ..
            } => format!("image:{id}:{image:?}:{aspect}:{}", points.len()),
        }
    }
}
#[derive(Resource, Default)]
pub struct Forms(pub BTreeMap<&'static str, Vec<Row>>);
#[derive(Resource, Default)]
pub struct StudioInput {
    pub pointer: bool,
    pub keyboard: bool,
}
#[derive(Component, Clone, Default)]
struct Binding(String);
#[derive(Component)]
struct Form(&'static str);
#[derive(Component)]
struct FormLabel(usize);
#[derive(Component)]
struct Panel;
#[derive(Component)]
struct Preview;
#[derive(Component)]
struct Overlay(usize);
#[derive(Resource, Default)]
struct ImageDrag(Option<Entity>);
#[derive(Resource, Default)]
struct Structures(BTreeMap<&'static str, Vec<String>>);

#[derive(Resource, Default)]
struct StudioFont(Handle<Font>);

pub struct StudioUiPlugin;
impl Plugin for StudioUiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(FeathersPlugins)
            .insert_resource(UiTheme(create_dark_theme()))
            .init_resource::<ImageDrag>()
            .init_resource::<StudioFont>()
            .init_resource::<Forms>()
            .init_resource::<Structures>()
            .init_resource::<StudioInput>()
            .add_message::<ControlEvent>()
            .add_systems(Startup, setup)
            .add_systems(
                PreUpdate,
                capture_input.after(bevy::picking::PickingSystems::Last),
            )
            .add_systems(
                PostUpdate,
                (render_forms, image_drag, update_overlays, scroll).chain(),
            );
    }
}
fn setup(mut commands: Commands, assets: Res<AssetServer>) {
    commands.insert_resource(StudioFont(
        assets.load(bevy::feathers::constants::fonts::REGULAR),
    ));
    let panel = commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(0.),
                top: Val::Px(0.),
                width: Val::Px(360.),
                max_width: Val::Percent(100.),
                height: Val::Percent(100.),
                flex_direction: FlexDirection::Column,
                overflow: Overflow::scroll_y(),
                padding: UiRect::all(Val::Px(12.)),
                row_gap: Val::Px(8.),
                ..default()
            },
            BackgroundColor(Color::srgb(0.055, 0.065, 0.085)),
            GlobalZIndex(20),
            ScrollPosition::default(),
            Panel,
            TabGroup::default(),
        ))
        .id();
    for name in ["motion", "anny"] {
        commands.spawn((
            Node {
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.),
                flex_shrink: 0.,
                ..default()
            },
            Form(name),
            ChildOf(panel),
        ));
    }
}
pub(crate) fn capture_input(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut focus: ResMut<InputFocus>,
    buttons: Res<ButtonInput<MouseButton>>,
    edits: Query<(), With<EditableText>>,
    mut input: ResMut<StudioInput>,
) {
    input.pointer = windows.single().is_ok_and(|w| {
        !w.focused
            || w.cursor_position()
                .is_some_and(|p| p.x < 360. && p.y >= 0. && p.y < w.height())
    });
    if buttons.just_pressed(MouseButton::Left) && !input.pointer {
        focus.clear();
    }
    input.keyboard = focus.get().is_some_and(|e| edits.contains(e));
}
fn scroll(
    mut wheel: MessageReader<MouseWheel>,
    input: Res<StudioInput>,
    mut panels: Query<(&mut ScrollPosition, &ComputedNode), With<Panel>>,
) {
    let amount: f32 = wheel
        .read()
        .map(|w| {
            w.y * if w.unit == MouseScrollUnit::Line {
                30.
            } else {
                1.
            }
        })
        .sum();
    if input.pointer {
        for (mut panel, node) in &mut panels {
            panel.0.y = (panel.0.y - amount).clamp(
                0.,
                (node.content_size().y - node.size().y).max(0.) * node.inverse_scale_factor(),
            );
        }
    }
}
fn text_event(
    event: On<TextEditChange>,
    bindings: Query<(&Binding, &EditableText)>,
    mut events: MessageWriter<ControlEvent>,
) {
    if let Ok((key, text)) = bindings.get(event.event_target()) {
        events.write(ControlEvent {
            id: key.0.clone(),
            value: Value::Text(text.value().to_string()),
        });
    }
}
fn button_event(
    event: On<Activate>,
    bindings: Query<&Binding>,
    mut events: MessageWriter<ControlEvent>,
) {
    if let Ok(key) = bindings.get(event.event_target()) {
        events.write(ControlEvent {
            id: key.0.clone(),
            value: Value::Action,
        });
    }
}
fn slider_event(
    event: On<ValueChange<f32>>,
    bindings: Query<&Binding>,
    mut commands: Commands,
    mut events: MessageWriter<ControlEvent>,
) {
    if let Ok(key) = bindings.get(event.event_target()) {
        commands
            .entity(event.event_target())
            .insert(SliderValue(event.value));
        events.write(ControlEvent {
            id: key.0.clone(),
            value: Value::Number(event.value),
        });
    }
}
fn toggle_event(
    event: On<ValueChange<bool>>,
    bindings: Query<&Binding>,
    mut events: MessageWriter<ControlEvent>,
) {
    if let Ok(key) = bindings.get(event.event_target()) {
        events.write(ControlEvent {
            id: key.0.clone(),
            value: Value::Toggle(event.value),
        });
    }
}
fn select_event(
    event: On<ValueChange<Entity>>,
    bindings: Query<&Binding>,
    indices: Query<&OptionIndex>,
    mut events: MessageWriter<ControlEvent>,
) {
    if let (Ok(key), Ok(index)) = (bindings.get(event.event_target()), indices.get(event.value)) {
        events.write(ControlEvent {
            id: key.0.clone(),
            value: Value::Choice(index.0),
        });
    }
}
fn label(commands: &mut Commands, parent: Entity, text: &str, font: &Handle<Font>) {
    commands.spawn((
        Text::new(text),
        TextFont {
            font: font.clone().into(),
            font_size: FontSize::Px(13.),
            ..default()
        },
        TextColor(Color::srgb(0.85, 0.87, 0.92)),
        ChildOf(parent),
    ));
}
#[allow(clippy::too_many_arguments)]
fn render_forms(
    mut commands: Commands,
    forms: Res<Forms>,
    mut structures: ResMut<Structures>,
    font: Res<StudioFont>,
    roots: Query<(Entity, &Form, Option<&Children>)>,
    sliders: Query<(Entity, &Binding, &SliderValue)>,
    mut edits: Query<(Entity, &Binding, &mut EditableText)>,
    bindings: Query<(Entity, &Binding, Has<InteractionDisabled>, Has<Checked>)>,
    mut labels: Query<(&FormLabel, &ChildOf, &mut Text)>,
    focus: Res<InputFocus>,
) {
    for (root, form, children) in &roots {
        let rows = forms.0.get(form.0).map(Vec::as_slice).unwrap_or_default();
        let structure: Vec<_> = rows.iter().map(Row::structure).collect();
        if structures.0.get(form.0) != Some(&structure) {
            structures.0.insert(form.0, structure);
            for (index, row) in rows.iter().enumerate() {
                let parent = commands
                    .spawn((
                        Node {
                            flex_direction: FlexDirection::Column,
                            row_gap: Val::Px(3.),
                            flex_shrink: 0.,
                            ..default()
                        },
                        ChildOf(root),
                    ))
                    .id();
                match row {
                    Row::Label(text) => {
                        commands.spawn((
                            Text::new(text),
                            TextFont {
                                font: font.0.clone().into(),
                                font_size: FontSize::Px(13.),
                                ..default()
                            },
                            FormLabel(index),
                            ChildOf(root),
                        ));
                        commands.entity(parent).despawn();
                    }
                    Row::Buttons(items) => {
                        commands.entity(parent).insert(Node {
                            flex_direction: FlexDirection::Row,
                            flex_wrap: FlexWrap::Wrap,
                            column_gap: Val::Px(4.),
                            row_gap: Val::Px(4.),
                            ..default()
                        });
                        for (id, title, enabled) in items {
                            let id = id.clone();
                            let title = title.clone();
                            let mut button=commands.spawn_scene(bsn!{ @FeathersButton { @caption:bsn!{@caption(title)} } Binding(id) ChildOf(parent) on(button_event) });
                            if !enabled {
                                button.insert(InteractionDisabled);
                            }
                        }
                    }
                    Row::Button { id, label, enabled } => {
                        let id = id.clone();
                        let title = label.clone();
                        let mut button=commands.spawn_scene(bsn! { @FeathersButton { @caption: bsn!{ @caption(title) } } Binding(id) ChildOf(parent) on(button_event) });
                        if !enabled {
                            button.insert(InteractionDisabled);
                        }
                    }
                    Row::Text {
                        id,
                        label: caption,
                        value,
                        multiline,
                    } => {
                        label(&mut commands, parent, caption, &font.0);
                        // Filtering changes select options while a text field
                        // is being edited. Keep the editor entity, cursor and
                        // focus; detach it before deleting the previous rows.
                        if let Some((entity, _, mut edit)) =
                            edits.iter_mut().find(|(_, key, _)| &key.0 == id)
                        {
                            if focus.get() != Some(entity) && edit.value().to_string() != *value {
                                edit.queue_edit(TextEdit::SelectAll);
                                edit.queue_edit(TextEdit::Insert(value.clone().into()));
                            }
                            commands.entity(entity).insert(ChildOf(parent));
                            continue;
                        }
                        commands
                            .spawn((
                                Node {
                                    width: Val::Percent(100.),
                                    padding: UiRect::all(Val::Px(6.)),
                                    border: UiRect::all(Val::Px(1.)),
                                    ..default()
                                },
                                BorderColor::all(Color::srgb(0.3, 0.36, 0.46)),
                                BackgroundColor(Color::srgb(0.025, 0.035, 0.05)),
                                TextInput,
                                EditableText {
                                    allow_newlines: *multiline,
                                    visible_lines: Some(if *multiline { 3. } else { 1. }),
                                    visible_width: Some(26.),
                                    max_characters: Some(4096),
                                    ..EditableText::new(value)
                                },
                                TextFont {
                                    font: font.0.clone().into(),
                                    font_size: FontSize::Px(14.),
                                    ..default()
                                },
                                TextCursorStyle::default(),
                                TabIndex(0),
                                Binding(id.clone()),
                                ChildOf(parent),
                            ))
                            .observe(text_event);
                    }
                    Row::Slider {
                        id,
                        label: caption,
                        value,
                        min,
                        max,
                        step,
                    } => {
                        label(&mut commands, parent, caption, &font.0);
                        let id = id.clone();
                        let (min, max, value, step) = (*min, *max, *value, *step);
                        commands.spawn_scene(bsn! { @FeathersSlider { @min:min,@max:max } SliderValue(value) SliderStep(step) SliderPrecision(2) Binding(id) ChildOf(parent) on(slider_event) });
                    }
                    Row::Toggle {
                        id,
                        label: toggle_caption,
                        value,
                    } => {
                        let id = id.clone();
                        let title = toggle_caption.clone();
                        let button=commands.spawn_scene(bsn! { @FeathersCheckbox { @caption: bsn!{ @caption(title) } } Binding(id) ChildOf(parent) on(toggle_event) }).id();
                        if *value {
                            commands.entity(button).insert(Checked);
                        }
                    }
                    Row::Select {
                        id,
                        label: caption,
                        options,
                        value,
                    } => {
                        label(&mut commands, parent, caption, &font.0);
                        let id = id.clone();
                        let options = list_rows_from_strings(options, Some(*value));
                        commands.spawn_scene(bsn! { @FeathersSelect { @options:options } Binding(id) ChildOf(parent) on(select_event) });
                    }
                    Row::Image {
                        id,
                        image,
                        aspect,
                        points,
                        ..
                    } => {
                        let size = Vec2::new((280. * aspect).min(336.), (336. / aspect).min(280.));
                        let preview = commands
                            .spawn((
                                Node {
                                    width: Val::Px(size.x),
                                    max_width: Val::Percent(100.),
                                    height: Val::Px(size.y),
                                    aspect_ratio: Some(*aspect),
                                    overflow: Overflow::clip(),
                                    ..default()
                                },
                                ImageNode::new(image.clone()),
                                Preview,
                                Binding(id.clone()),
                                ChildOf(parent),
                            ))
                            .id();
                        commands.spawn((
                            Node {
                                position_type: PositionType::Absolute,
                                border: UiRect::all(Val::Px(2.)),
                                ..default()
                            },
                            BorderColor::all(Color::srgb(1., 0.75, 0.15)),
                            Overlay(usize::MAX),
                            ChildOf(preview),
                        ));
                        for i in 0..points.len() {
                            commands.spawn((
                                Node {
                                    position_type: PositionType::Absolute,
                                    width: Val::Px(5.),
                                    height: Val::Px(5.),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.1, 1., 0.7)),
                                Overlay(i),
                                ChildOf(preview),
                            ));
                        }
                    }
                }
            }
            if let Some(children) = children {
                for child in children {
                    commands.entity(*child).despawn();
                }
            }
        } else {
            // Inputs remain the same entities while typing or dragging.
            for row in rows {
                match row {
                    Row::Slider { id, value, .. } => {
                        for (entity, key, slider) in &sliders {
                            if &key.0 == id && slider.0 != *value {
                                commands.entity(entity).insert(SliderValue(*value));
                            }
                        }
                    }
                    Row::Text { id, value, .. } => {
                        for (entity, key, mut edit) in &mut edits {
                            if &key.0 == id
                                && focus.get() != Some(entity)
                                && edit.value().to_string() != *value
                            {
                                edit.queue_edit(TextEdit::SelectAll);
                                edit.queue_edit(TextEdit::Insert(value.clone().into()));
                            }
                        }
                    }
                    Row::Buttons(items) => {
                        for (id, _, enabled) in items {
                            for (entity, key, disabled, _checked) in &bindings {
                                if &key.0 == id {
                                    if *enabled && disabled {
                                        commands.entity(entity).remove::<InteractionDisabled>();
                                    } else if !*enabled && !disabled {
                                        commands.entity(entity).insert(InteractionDisabled);
                                    }
                                }
                            }
                        }
                    }
                    Row::Button { id, enabled, .. } => {
                        for (entity, key, disabled, _checked) in &bindings {
                            if &key.0 == id {
                                if *enabled && disabled {
                                    commands.entity(entity).remove::<InteractionDisabled>();
                                } else if !*enabled && !disabled {
                                    commands.entity(entity).insert(InteractionDisabled);
                                }
                            }
                        }
                    }
                    Row::Toggle { id, value, .. } => {
                        for (entity, key, _disabled, checked) in &bindings {
                            if &key.0 == id {
                                if *value && !checked {
                                    commands.entity(entity).insert(Checked);
                                } else if !*value && checked {
                                    commands.entity(entity).remove::<Checked>();
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            for (slot, parent, mut text) in &mut labels {
                if parent.parent() == root
                    && let Some(Row::Label(value)) = rows.get(slot.0)
                    && text.0 != *value
                {
                    text.0 = value.clone();
                }
            }
        }
    }
}

// Coordinates stay normalized so DPI changes and scrolling cannot alter the crop.
fn image_drag(
    windows: Query<&Window, With<PrimaryWindow>>,
    buttons: Res<ButtonInput<MouseButton>>,
    previews: Query<(Entity, &Binding, &ComputedNode, &UiGlobalTransform), With<Preview>>,
    mut drag: ResMut<ImageDrag>,
    mut events: MessageWriter<ControlEvent>,
) {
    if buttons.just_released(MouseButton::Left) {
        drag.0 = None;
    }
    if !buttons.pressed(MouseButton::Left) {
        return;
    }
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    for (entity, binding, node, transform) in &previews {
        let Some(inverse) = transform.try_inverse() else {
            continue;
        };
        let physical = cursor * window.scale_factor();
        let local = inverse.transform_point2(physical) + node.size() / 2.;
        let unit = local / node.size();
        if buttons.just_pressed(MouseButton::Left)
            && unit.cmpge(Vec2::ZERO).all()
            && unit.cmple(Vec2::ONE).all()
        {
            drag.0 = Some(entity);
        }
        if drag.0 == Some(entity) {
            events.write(ControlEvent {
                id: binding.0.clone(),
                value: Value::Point(unit.clamp(Vec2::ZERO, Vec2::ONE)),
            });
        }
    }
}
fn update_overlays(
    forms: Res<Forms>,
    previews: Query<&Binding, With<Preview>>,
    mut overlays: Query<(&Overlay, &ChildOf, &mut Node)>,
) {
    for (overlay, parent, mut node) in &mut overlays {
        let Ok(binding) = previews.get(parent.parent()) else {
            continue;
        };
        let Some(Row::Image { crop, points, .. }) = forms
            .0
            .values()
            .flatten()
            .find(|row| matches!(row,Row::Image{id,..} if id==&binding.0))
        else {
            continue;
        };
        if overlay.0 == usize::MAX {
            let [left, top, width, height] = crop.map(|v| Val::Percent(v * 100.));
            if [node.left, node.top, node.width, node.height] != [left, top, width, height] {
                node.left = left;
                node.top = top;
                node.width = width;
                node.height = height;
            }
        } else if let Some(point) = points.get(overlay.0) {
            let left = Val::Percent(point[0] * 100.);
            let top = Val::Percent(point[1] * 100.);
            if node.left != left || node.top != top {
                node.left = left;
                node.top = top;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changing_form_options_preserves_text_focus_and_pending_edits() {
        let mut app = App::new();
        app.init_resource::<Forms>()
            .init_resource::<Structures>()
            .init_resource::<StudioFont>()
            .init_resource::<InputFocus>()
            .add_systems(Update, render_forms);
        let root = app.world_mut().spawn(Form("test")).id();
        app.world_mut()
            .resource_mut::<Forms>()
            .0
            .insert("test", vec![Row::text("filter", "Find joint", "ha")]);
        app.update();
        let editor = app
            .world_mut()
            .query_filtered::<Entity, With<EditableText>>()
            .single(app.world())
            .unwrap();
        let old_parent = app.world().get::<ChildOf>(editor).unwrap().parent();
        app.world_mut()
            .resource_mut::<InputFocus>()
            .set(editor, bevy::input_focus::FocusCause::Navigated);
        app.world_mut()
            .get_mut::<EditableText>(editor)
            .unwrap()
            .queue_edit(TextEdit::Insert("nd".into()));
        // Structural changes include the filtered dropdown's changing options.
        app.world_mut()
            .resource_mut::<Forms>()
            .0
            .insert("test", vec![Row::text("filter", "Matching joints", "ha")]);
        app.update();
        assert_eq!(app.world().resource::<InputFocus>().get(), Some(editor));
        let text = app.world().get::<EditableText>(editor).unwrap();
        assert_eq!(text.value().to_string(), "ha");
        assert_eq!(text.pending_edits.len(), 2); // initial cursor move plus insertion
        assert_ne!(
            app.world().get::<ChildOf>(editor).unwrap().parent(),
            old_parent
        );
        assert!(app.world().get_entity(old_parent).is_err());
        assert_eq!(
            app.world_mut()
                .query_filtered::<Entity, With<EditableText>>()
                .iter(app.world())
                .count(),
            1
        );
        assert!(app.world().get_entity(root).is_ok());
    }
}
