// This lint has lots of false positives ATM, see
// https://github.com/Manishearth/rust-clippy/issues/761
#![cfg_attr(feature = "cargo-clippy", allow(new_without_default))]
// False positives with images
#![cfg_attr(feature = "cargo-clippy", allow(doc_markdown))]

extern crate byteorder;
extern crate cast;
#[macro_use]
extern crate itertools;

use std::borrow::Cow;
use std::fs::File;
use std::io;
use std::path::Path;
use std::process::{Child, Command};
use std::str;

use data::Matrix;
use traits::{Configure, Set};

mod data;
mod display;
mod map;

pub mod axis;
pub mod candlestick;
pub mod curve;
pub mod errorbar;
pub mod filledcurve;
pub mod grid;
pub mod key;
pub mod prelude;
pub mod proxy;
pub mod traits;

/// Plot container
#[derive(Clone)]
pub struct Figure {
    alpha: Option<f64>,
    axes: map::axis::Map<axis::Properties>,
    box_width: Option<f64>,
    font: Option<Cow<'static, str>>,
    font_size: Option<f64>,
    key: Option<key::Properties>,
    output: Cow<'static, Path>,
    plots: Vec<Plot>,
    size: Option<(usize, usize)>,
    terminal: Terminal,
    tics: map::axis::Map<String>,
    title: Option<Cow<'static, str>>,
}

impl Figure {
    /// Creates an empty figure
    pub fn new() -> Figure {
        Figure {
            alpha: None,
            axes: map::axis::Map::new(),
            box_width: None,
            font: None,
            font_size: None,
            key: None,
            output: Cow::Borrowed(Path::new("output.plot")),
            plots: Vec::new(),
            size: None,
            terminal: Terminal::Svg,
            tics: map::axis::Map::new(),
            title: None,
        }
    }

    fn script(&self) -> Vec<u8> {
        let mut s = String::new();

        s.push_str(&format!("set output '{}'\n", self.output.display()));

        if let Some(width) = self.box_width {
            s.push_str(&format!("set boxwidth {}\n", width))
        }

        if let Some(ref title) = self.title {
            s.push_str(&format!("set title '{}'\n", title))
        }

        for axis in self.axes.iter() {
            s.push_str(&axis.script());
        }

        for (_, script) in self.tics.iter() {
            s.push_str(script);
        }

        if let Some(ref key) = self.key {
            s.push_str(&key.script())
        }

        if let Some(alpha) = self.alpha {
            s.push_str(&format!("set style fill transparent solid {}\n", alpha))
        }

        s.push_str(&format!("set terminal {} dashed", self.terminal.display()));

        if let Some((width, height)) = self.size {
            s.push_str(&format!(" size {}, {}", width, height))
        }

        if let Some(ref name) = self.font {
            if let Some(size) = self.font_size {
                s.push_str(&format!(" font '{},{}'", name, size))
            } else {
                s.push_str(&format!(" font '{}'", name))
            }
        }

        // TODO This removes the crossbars from the ends of error bars, but should be configurable
        s.push_str("\nunset bars\n");

        let mut is_first_plot = true;
        for plot in &self.plots {
            let data = plot.data();

            if data.bytes().is_empty() {
                continue;
            }

            if is_first_plot {
                s.push_str("plot ");
                is_first_plot = false;
            } else {
                s.push_str(", ");
            }

            s.push_str(&format!(
                "'-' binary endian=little record={} format='%float64' using ",
                data.nrows()
            ));

            let mut is_first_col = true;
            for col in 0..data.ncols() {
                if is_first_col {
                    is_first_col = false;
                } else {
                    s.push(':');
                }
                s.push_str(&(col + 1).to_string());
            }
            s.push(' ');

            s.push_str(plot.script());
        }

        let mut buffer = s.into_bytes();
        let mut is_first = true;
        for plot in &self.plots {
            if is_first {
                is_first = false;
                buffer.push(b'\n');
            }
            buffer.extend_from_slice(plot.data().bytes());
        }

        buffer
    }

    /// Spawns a drawing child process
    ///
    /// NOTE: stderr, stdin, and stdout are piped
    pub fn draw(&mut self) -> io::Result<Child> {
        use std::process::Stdio;

        let mut gnuplot = try!{
            Command::new("gnuplot").
                stderr(Stdio::piped()).
                stdin(Stdio::piped()).
                stdout(Stdio::piped()).
                spawn()
        };
        try!(self.dump(gnuplot.stdin.as_mut().unwrap()));
        Ok(gnuplot)
    }

    /// Dumps the script required to produce the figure into `sink`
    pub fn dump<W>(&mut self, sink: &mut W) -> io::Result<&mut Figure>
    where
        W: io::Write,
    {
        try!(sink.write_all(&self.script()));
        Ok(self)
    }

    /// Saves the script required to produce the figure to `path`
    pub fn save(&self, path: &Path) -> io::Result<&Figure> {
        use std::io::Write;

        try!((try!(File::create(path))).write_all(&self.script()));
        Ok(self)
    }
}

impl Configure<Axis> for Figure {
    type Properties = axis::Properties;

    /// Configures an axis
    fn configure<F>(&mut self, axis: Axis, configure: F) -> &mut Figure
    where
        F: FnOnce(&mut axis::Properties) -> &mut axis::Properties,
    {
        if self.axes.contains_key(axis) {
            configure(self.axes.get_mut(axis).unwrap());
        } else {
            let mut properties = Default::default();
            configure(&mut properties);
            self.axes.insert(axis, properties);
        }
        self
    }
}

impl Configure<Key> for Figure {
    type Properties = key::Properties;

    /// Configures the key (legend)
    fn configure<F>(&mut self, _: Key, configure: F) -> &mut Figure
    where
        F: FnOnce(&mut key::Properties) -> &mut key::Properties,
    {
        if self.key.is_some() {
            configure(self.key.as_mut().unwrap());
        } else {
            let mut key = Default::default();
            configure(&mut key);
            self.key = Some(key);
        }
        self
    }
}

impl Set<BoxWidth> for Figure {
    /// Changes the box width of all the box related plots (bars, candlesticks, etc)
    ///
    /// **Note** The default value is 0
    ///
    /// # Panics
    ///
    /// Panics if `width` is a negative value
    fn set(&mut self, width: BoxWidth) -> &mut Figure {
        let width = width.0;

        assert!(width >= 0.);

        self.box_width = Some(width);
        self
    }
}

impl Set<Font> for Figure {
    /// Changes the font
    fn set(&mut self, font: Font) -> &mut Figure {
        self.font = Some(font.0);
        self
    }
}

impl Set<FontSize> for Figure {
    /// Changes the size of the font
    ///
    /// # Panics
    ///
    /// Panics if `size` is a non-positive value
    fn set(&mut self, size: FontSize) -> &mut Figure {
        let size = size.0;

        assert!(size >= 0.);

        self.font_size = Some(size);
        self
    }
}

impl Set<Output> for Figure {
    /// Changes the output file
    ///
    /// **Note** The default output file is `output.plot`
    fn set(&mut self, output: Output) -> &mut Figure {
        self.output = output.0;
        self
    }
}

impl Set<Size> for Figure {
    /// Changes the figure size
    fn set(&mut self, size: Size) -> &mut Figure {
        self.size = Some((size.0, size.1));
        self
    }
}

impl Set<Terminal> for Figure {
    /// Changes the output terminal
    ///
    /// **Note** By default, the terminal is set to `Svg`
    fn set(&mut self, terminal: Terminal) -> &mut Figure {
        self.terminal = terminal;
        self
    }
}

impl Set<Title> for Figure {
    /// Sets the title
    fn set(&mut self, title: Title) -> &mut Figure {
        self.title = Some(title.0);
        self
    }
}

impl Default for Figure {
    fn default() -> Self {
        Self::new()
    }
}

/// Box width for box-related plots: bars, candlesticks, etc
#[derive(Clone, Copy)]
pub struct BoxWidth(pub f64);

/// A font name
pub struct Font(Cow<'static, str>);

/// The size of a font
#[derive(Clone, Copy)]
pub struct FontSize(pub f64);

/// The key or legend
#[derive(Clone, Copy)]
pub struct Key;

/// Plot label
pub struct Label(Cow<'static, str>);

/// Width of the lines
#[derive(Clone, Copy)]
pub struct LineWidth(pub f64);

/// Fill color opacity
#[derive(Clone, Copy)]
pub struct Opacity(pub f64);

/// Output file path
pub struct Output(Cow<'static, Path>);

/// Size of the points
#[derive(Clone, Copy)]
pub struct PointSize(pub f64);

/// Axis range
#[derive(Clone, Copy)]
pub enum Range {
    /// Autoscale the axis
    Auto,
    /// Set the limits of the axis
    Limits(f64, f64),
}

/// Figure size
#[derive(Clone, Copy)]
pub struct Size(pub usize, pub usize);

/// Labels attached to the tics of an axis
pub struct TicLabels<P, L> {
    /// Labels to attach to the tics
    pub labels: L,
    /// Position of the tics on the axis
    pub positions: P,
}

/// Figure title
pub struct Title(Cow<'static, str>);

/// A pair of axes that define a coordinate system
#[allow(missing_docs)]
#[derive(Clone, Copy)]
pub enum Axes {
    BottomXLeftY,
    BottomXRightY,
    TopXLeftY,
    TopXRightY,
}

/// A coordinate axis
#[derive(Clone, Copy)]
pub enum Axis {
    /// X axis on the bottom side of the figure
    BottomX,
    /// Y axis on the left side of the figure
    LeftY,
    /// Y axis on the right side of the figure
    RightY,
    /// X axis on the top side of the figure
    TopX,
}

impl Axis {
    fn next(&self) -> Option<Axis> {
        use Axis::*;

        match *self {
            BottomX => Some(LeftY),
            LeftY => Some(RightY),
            RightY => Some(TopX),
            TopX => None,
        }
    }
}

/// Color
#[allow(missing_docs)]
#[derive(Clone, Copy)]
pub enum Color {
    Black,
    Blue,
    Cyan,
    DarkViolet,
    ForestGreen,
    Gold,
    Gray,
    Green,
    Magenta,
    Red,
    /// Custom RGB color
    Rgb(u8, u8, u8),
    White,
    Yellow,
}

/// Grid line
#[derive(Clone, Copy)]
pub enum Grid {
    /// Major gridlines
    Major,
    /// Minor gridlines
    Minor,
}

impl Grid {
    fn next(&self) -> Option<Grid> {
        use Grid::*;

        match *self {
            Major => Some(Minor),
            Minor => None,
        }
    }
}

/// Line type
#[allow(missing_docs)]
#[derive(Clone, Copy)]
pub enum LineType {
    Dash,
    Dot,
    DotDash,
    DotDotDash,
    /// Line made of minimally sized dots
    SmallDot,
    Solid,
}

/// Point type
#[allow(missing_docs)]
#[derive(Clone, Copy)]
pub enum PointType {
    Circle,
    FilledCircle,
    FilledSquare,
    FilledTriangle,
    Plus,
    Square,
    Star,
    Triangle,
    X,
}

/// Axis scale
#[allow(missing_docs)]
#[derive(Clone, Copy)]
pub enum Scale {
    Linear,
    Logarithmic,
}

/// Axis scale factor
#[allow(missing_docs)]
#[derive(Clone, Copy)]
pub struct ScaleFactor(pub f64);

/// Output terminal
#[allow(missing_docs)]
#[derive(Clone, Copy)]
pub enum Terminal {
    Svg,
}

/// Not public version of `std::default::Default`, used to not leak default constructors into the
/// public API
trait Default {
    /// Creates `Properties` with default configuration
    fn default() -> Self;
}

/// Enums that can produce gnuplot code
trait Display<S> {
    /// Translates the enum in gnuplot code
    fn display(&self) -> S;
}

/// Curve variant of Default
trait CurveDefault<S> {
    /// Creates `curve::Properties` with default configuration
    fn default(S) -> Self;
}

/// Error bar variant of Default
trait ErrorBarDefault<S> {
    /// Creates `errorbar::Properties` with default configuration
    fn default(S) -> Self;
}

/// Structs that can produce gnuplot code
trait Script {
    /// Translates some configuration struct into gnuplot code
    fn script(&self) -> String;
}

#[derive(Clone)]
struct Plot {
    data: Matrix,
    script: String,
}

impl Plot {
    fn new<S>(data: Matrix, script: &S) -> Plot
    where
        S: Script,
    {
        Plot {
            data: data,
            script: script.script(),
        }
    }

    fn data(&self) -> &Matrix {
        &self.data
    }

    fn script(&self) -> &str {
        &self.script
    }
}
/// Returns `gnuplot` version
// FIXME Parsing may fail
pub fn version() -> io::Result<(usize, usize, usize)> {
    let stdout = try!(Command::new("gnuplot").arg("--version").output()).stdout;
    let mut words = str::from_utf8(&stdout).unwrap().split_whitespace().skip(1);
    let mut version = words.next().unwrap().split('.');
    let major = version.next().unwrap().parse().unwrap();
    let minor = version.next().unwrap().parse().unwrap();
    let patchlevel = words.nth(1).unwrap().parse().unwrap();

    Ok((major, minor, patchlevel))
}

fn scale_factor(map: &map::axis::Map<axis::Properties>, axes: Axes) -> (f64, f64) {
    use Axes::*;
    use Axis::*;

    match axes {
        BottomXLeftY => (
            map.get(BottomX).map_or(1., |props| props.scale_factor()),
            map.get(LeftY).map_or(1., |props| props.scale_factor()),
        ),
        BottomXRightY => (
            map.get(BottomX).map_or(1., |props| props.scale_factor()),
            map.get(RightY).map_or(1., |props| props.scale_factor()),
        ),
        TopXLeftY => (
            map.get(TopX).map_or(1., |props| props.scale_factor()),
            map.get(LeftY).map_or(1., |props| props.scale_factor()),
        ),
        TopXRightY => (
            map.get(TopX).map_or(1., |props| props.scale_factor()),
            map.get(RightY).map_or(1., |props| props.scale_factor()),
        ),
    }
}

// XXX :-1: to intra-crate privacy rules
/// Private
trait ScaleFactorTrait {
    /// Private
    fn scale_factor(&self) -> f64;
}

#[cfg(test)]
mod test {
    #[test]
    fn version() {
        if let Ok(version) = super::version() {
            let (major, _, _) = version;
            assert!(major >= 4);
        } else {
            println!("Gnuplot not installed.");
        }
    }
}
