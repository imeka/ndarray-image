use std::ops::{Add, Sub};

use ndarray::{s, Array, Array2, ArrayBase, ArrayViewMut1, Data, Ix3, Zip};
use num_traits::{FromPrimitive, Num, ToPrimitive};

use crate::{array_like, pad, round_ties_even, spline_filter, BorderMode, PadMode};

/// Shift an array.
///
/// The array is shifted using spline interpolation of the requested order. Points outside the
/// boundaries of the input are filled according to the given mode.
///
/// * `data` - A 3D array of the data to shift.
/// * `shift` - The shift along the axes.
/// * `order` - The order of the spline.
/// * `mode` - The mode parameter determines how the input array is extended beyond its boundaries.
/// * `prefilter` - Determines if the input array is prefiltered with spline_filter before
///   interpolation. The default is `true`, which will create a temporary `f64` array of filtered
///   values if `order > 1`. If setting this to `false`, the output will be slightly blurred if
///   `order > 1`, unless the input is prefiltered.
pub fn shift<S, A>(
    data: &ArrayBase<S, Ix3>,
    shift: [f64; 3],
    order: usize,
    mode: BorderMode<A>,
    prefilter: bool,
) -> Array<A, Ix3>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd + ToPrimitive,
{
    let dim = [data.dim().0, data.dim().1, data.dim().2];
    let shift = shift.map(|s| -s);
    run_zoom_shift(data, dim, [1.0, 1.0, 1.0], shift, order, mode, prefilter)
}

/// Zoom an array.
///
/// The array is zoomed using spline interpolation of the requested order.
///
/// * `data` - A 3D array of the data to zoom
/// * `zoom` - The zoom factor along the axes.
/// * `order` - The order of the spline.
/// * `mode` - The mode parameter determines how the input array is extended beyond its boundaries.
/// * `prefilter` - Determines if the input array is prefiltered with spline_filter before
///   interpolation. The default is `true`, which will create a temporary `f64` array of filtered
///   values if `order > 1`. If setting this to `false`, the output will be slightly blurred if
///   `order > 1`, unless the input is prefiltered.
pub fn zoom<S, A>(
    data: &ArrayBase<S, Ix3>,
    zoom: [f64; 3],
    order: usize,
    mode: BorderMode<A>,
    prefilter: bool,
) -> Array<A, Ix3>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd + ToPrimitive,
{
    let mut o_dim = data.raw_dim();
    for (ax, (&ax_len, zoom)) in data.shape().iter().zip(zoom.iter()).enumerate() {
        o_dim[ax] = round_ties_even(ax_len as f64 * zoom) as usize;
    }
    let o_dim = [o_dim[0], o_dim[1], o_dim[2]];

    let mut nom = data.raw_dim();
    let mut div = o_dim.clone();
    for ax in 0..data.ndim() {
        nom[ax] -= 1;
        div[ax] -= 1;
    }
    let zoom = [
        nom[0] as f64 / div[0] as f64,
        nom[1] as f64 / div[1] as f64,
        nom[2] as f64 / div[2] as f64,
    ];

    run_zoom_shift(data, o_dim, zoom, [0.0, 0.0, 0.0], order, mode, prefilter)
}

fn run_zoom_shift<S, A>(
    data: &ArrayBase<S, Ix3>,
    odim: [usize; 3],
    zooms: [f64; 3],
    shifts: [f64; 3],
    order: usize,
    mode: BorderMode<A>,
    prefilter: bool,
) -> Array<A, Ix3>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + PartialOrd + ToPrimitive,
{
    let idim = [data.dim().0, data.dim().1, data.dim().2];
    let mut out = array_like(&data, odim, A::zero());
    if prefilter && order > 1 {
        // We need to allocate and work on filtered data
        let (data, nb_prepad) = match mode {
            BorderMode::Nearest => {
                let padded = pad(data, &[[12, 12]], PadMode::Edge);
                (spline_filter(&padded, order, mode), 12)
            }
            _ => (spline_filter(data, order, mode), 0),
        };
        let reslicer = ZoomShiftReslicer::new(idim, odim, zooms, shifts, order, mode, nb_prepad);
        Zip::indexed(&mut out).for_each(|idx, o| {
            *o = A::from_f64(reslicer.interpolate(&data, idx)).unwrap();
        });
    } else {
        // We can use the &data as-is
        let reslicer = ZoomShiftReslicer::new(idim, odim, zooms, shifts, order, mode, 0);
        Zip::indexed(&mut out).for_each(|idx, o| {
            *o = A::from_f64(reslicer.interpolate(data, idx)).unwrap();
        });
    }
    out
}

/// Zoom shift transformation (only scaling and translation).
struct ZoomShiftReslicer {
    order: usize,
    offsets: [Vec<isize>; 3],
    edge_offsets: [Array2<isize>; 3],
    is_edge_case: [Vec<bool>; 3],
    splvals: [Array2<f64>; 3],
    zeros: [Vec<bool>; 3],
    cval: f64,
}

impl ZoomShiftReslicer {
    /// Build all necessary data to call `interpolate`.
    pub fn new<A>(
        idim: [usize; 3],
        odim: [usize; 3],
        zooms: [f64; 3],
        shifts: [f64; 3],
        order: usize,
        mode: BorderMode<A>,
        nb_prepad: isize,
    ) -> ZoomShiftReslicer
    where
        A: Copy + ToPrimitive,
    {
        let offsets = [vec![0; odim[0]], vec![0; odim[1]], vec![0; odim[2]]];
        let is_edge_case = [vec![false; odim[0]], vec![false; odim[1]], vec![false; odim[2]]];
        let (edge_offsets, splvals) = if order > 0 {
            let dim0 = (odim[0], order + 1);
            let dim1 = (odim[1], order + 1);
            let dim2 = (odim[2], order + 1);
            let e = [Array2::zeros(dim0), Array2::zeros(dim1), Array2::zeros(dim2)];
            let s = [Array2::zeros(dim0), Array2::zeros(dim1), Array2::zeros(dim2)];
            (e, s)
        } else {
            // We do not need to allocate when order == 0
            let e = [Array2::zeros((0, 0)), Array2::zeros((0, 0)), Array2::zeros((0, 0))];
            let s = [Array2::zeros((0, 0)), Array2::zeros((0, 0)), Array2::zeros((0, 0))];
            (e, s)
        };
        let zeros = [vec![false; odim[0]], vec![false; odim[1]], vec![false; odim[2]]];
        let cval = match mode {
            BorderMode::Constant(cval) => cval.to_f64().unwrap(),
            _ => 0.0,
        };

        let mut reslicer =
            ZoomShiftReslicer { order, offsets, edge_offsets, is_edge_case, splvals, zeros, cval };
        reslicer.build_arrays(idim, odim, zooms, shifts, order, mode, nb_prepad);
        reslicer
    }

    fn build_arrays<A>(
        &mut self,
        idim: [usize; 3],
        odim: [usize; 3],
        zooms: [f64; 3],
        shifts: [f64; 3],
        order: usize,
        mode: BorderMode<A>,
        nb_prepad: isize,
    ) where
        A: Copy,
    {
        // Modes without an anlaytic prefilter or explicit prepadding use mirror extension
        let spline_mode = match mode {
            BorderMode::Constant(_) | BorderMode::Wrap => BorderMode::Mirror,
            _ => mode,
        };
        let iorder = order as isize;
        let idim = [
            idim[0] as isize + 2 * nb_prepad,
            idim[1] as isize + 2 * nb_prepad,
            idim[2] as isize + 2 * nb_prepad,
        ];
        let nb_prepad = nb_prepad as f64;

        for axis in 0..3 {
            let splvals = &mut self.splvals[axis];
            let offsets = &mut self.offsets[axis];
            let edge_offsets = &mut self.edge_offsets[axis];
            let is_edge_case = &mut self.is_edge_case[axis];
            let zeros = &mut self.zeros[axis];
            let len = idim[axis] as f64;
            for from in 0..odim[axis] {
                let mut to = (from as f64 + shifts[axis]) * zooms[axis] + nb_prepad;
                match mode {
                    BorderMode::Nearest => {}
                    _ => to = map_coordinates(to, idim[axis] as f64, mode),
                };
                if to > -1.0 {
                    if order > 0 {
                        build_splines(to, &mut splvals.row_mut(from), order);
                    }
                    if order & 1 == 0 {
                        to += 0.5;
                    }

                    let start = to.floor() as isize - iorder / 2;
                    offsets[from] = start;
                    if start < 0 || start + iorder >= idim[axis] {
                        is_edge_case[from] = true;
                        for o in 0..=order {
                            let x = (start + o as isize) as f64;
                            let idx = map_coordinates(x, len, spline_mode) as isize;
                            edge_offsets[(from, o)] = idx - start;
                        }
                    }
                } else {
                    zeros[from] = true;
                }
            }
        }
    }

    /// Spline interpolation with up-to 8 neighbors of a point.
    pub fn interpolate<A, S>(&self, data: &ArrayBase<S, Ix3>, start: (usize, usize, usize)) -> f64
    where
        S: Data<Elem = A>,
        A: ToPrimitive + Add<Output = A> + Sub<Output = A> + Copy,
    {
        if self.zeros[0][start.0] || self.zeros[1][start.1] || self.zeros[2][start.2] {
            return self.cval;
        }

        // Order = 0
        // We do not want to go further because
        // - it would be uselessly slower
        // - self.splvals is empty so it would crash (although we could fill it with 1.0)
        if self.edge_offsets[0].is_empty() {
            let x = self.offsets[0][start.0] as usize;
            let y = self.offsets[1][start.1] as usize;
            let z = self.offsets[2][start.2] as usize;
            return data[(x, y, z)].to_f64().unwrap();
        }

        // Linear interpolation use a nxnxn block. This is simple enough, but we must adjust this
        // block when the `start` is near the edges.
        let n = self.order + 1;
        let valid_index = |original_offset, is_edge, start, d: usize, v| {
            (original_offset + if is_edge { self.edge_offsets[d][(start, v)] } else { v as isize })
                as usize
        };

        let original_offset_x = self.offsets[0][start.0];
        let is_edge_x = self.is_edge_case[0][start.0];
        let mut xs = [0; 6];
        let original_offset_y = self.offsets[1][start.1];
        let is_edge_y = self.is_edge_case[1][start.1];
        let mut ys = [0; 6];
        let original_offset_z = self.offsets[2][start.2];
        let is_edge_z = self.is_edge_case[2][start.2];
        let mut zs = [0; 6];
        for i in 0..n {
            xs[i] = valid_index(original_offset_x, is_edge_x, start.0, 0, i);
            ys[i] = valid_index(original_offset_y, is_edge_y, start.1, 1, i);
            zs[i] = valid_index(original_offset_z, is_edge_z, start.2, 2, i);
        }

        let mut t = 0.0;
        for (z, &idx_z) in zs[..n].iter().enumerate() {
            let spline_z = self.splvals[2][(start.2, z)];
            for (y, &idx_y) in ys[..n].iter().enumerate() {
                let spline_yz = self.splvals[1][(start.1, y)] * spline_z;
                for (x, &idx_x) in xs[..n].iter().enumerate() {
                    let spline_xyz = self.splvals[0][(start.0, x)] * spline_yz;
                    t += data[(idx_x, idx_y, idx_z)].to_f64().unwrap() * spline_xyz;
                }
            }
        }
        t
    }
}

fn build_splines(to: f64, spline: &mut ArrayViewMut1<f64>, order: usize) {
    let x = to - if order & 1 == 1 { to } else { to + 0.5 }.floor();
    match order {
        1 => spline[0] = 1.0 - x,
        2 => {
            spline[0] = 0.5 * (0.5 - x).powi(2);
            spline[1] = 0.75 - x * x;
        }
        3 => {
            let z = 1.0 - x;
            spline[0] = z * z * z / 6.0;
            spline[1] = (x * x * (x - 2.0) * 3.0 + 4.0) / 6.0;
            spline[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
        }
        4 => {
            let t = x * x;
            let y = 1.0 + x;
            let z = 1.0 - x;
            spline[0] = (0.5 - x).powi(4) / 24.0;
            spline[1] = y * (y * (y * (5.0 - y) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0;
            spline[2] = t * (t * 0.25 - 0.625) + 115.0 / 192.0;
            spline[3] = z * (z * (z * (5.0 - z) / 6.0 - 1.25) + 5.0 / 24.0) + 55.0 / 96.0;
        }
        5 => {
            let y = 1.0 - x;
            let t = y * y;
            spline[0] = y * t * t / 120.0;
            let y = x + 1.0;
            spline[1] = y * (y * (y * (y * (y / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425;
            let t = x * x;
            spline[2] = t * (t * (0.25 - x / 12.0) - 0.5) + 0.55;
            let z = 1.0 - x;
            let t = z * z;
            spline[3] = t * (t * (0.25 - z / 12.0) - 0.5) + 0.55;
            let z = z + 1.0;
            spline[4] = z * (z * (z * (z * (z / 24.0 - 0.375) + 1.25) - 1.75) + 0.625) + 0.425;
        }
        _ => panic!("order must be between 1 and 5"),
    }
    spline[order] = 1.0 - spline.slice(s![..order]).sum();
}

fn map_coordinates<A>(mut idx: f64, len: f64, mode: BorderMode<A>) -> f64 {
    match mode {
        BorderMode::Constant(_) => {
            if idx < 0.0 || idx >= len {
                idx = -1.0;
            }
        }
        BorderMode::Nearest => {
            if idx < 0.0 {
                idx = 0.0;
            } else if idx >= len {
                idx = len - 1.0;
            }
        }
        BorderMode::Mirror => {
            let s2 = 2.0 * len - 2.0;
            if idx < 0.0 {
                idx = s2 * (-idx / s2).floor() + idx;
                idx = if idx <= 1.0 - len { idx + s2 } else { -idx };
            } else if idx >= len {
                idx -= s2 * (idx / s2).floor();
                if idx >= len {
                    idx = s2 - idx;
                }
            }
        }
        BorderMode::Reflect => {
            let s2 = 2.0 * len;
            if idx < 0.0 {
                if idx < -s2 {
                    idx = s2 * (-idx / s2).floor() + idx;
                }
                idx = if idx < -len { idx + s2 } else { -idx - 1.0 };
            } else if idx >= len {
                idx -= s2 * (idx / s2).floor();
                if idx >= len {
                    idx = s2 - idx - 1.0;
                }
            }
        }
        BorderMode::Wrap => {
            let s = len - 1.0;
            if idx < 0.0 {
                idx += s * ((-idx / s).floor() + 1.0);
            } else if idx >= len {
                idx -= s * (idx / s).floor();
            }
        }
    };
    idx
}
