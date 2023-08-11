use std::ops::{Add, Sub};

use ndarray::{Array, Array2, ArrayBase, Data, Ix3, Zip};
use num_traits::{FromPrimitive, Num, ToPrimitive};

use crate::{array_like, spline_filter};

/// Zoom an array.
///
/// The array is zoomed using spline interpolation of the requested order.
///
/// * `data` - A 3D array of the data to zoom
/// * `zoom` - Number of values padded to the edges of each axis.
/// * `order` - The order of the spline.
/// * `mode` - The mode parameter determines how the input array is extended beyond its boundaries.
/// * `prefilter` - Determines if the input array is prefiltered with spline_filter before
///   interpolation. The default is `true`, which will create a temporary `f64` array of filtered
///   values if `order > 1`. If setting this to `false`, the output will be slightly blurred if
///   `order > 1`, unless the input is prefiltered.
pub fn zoom<S, A>(data: &ArrayBase<S, Ix3>, zoom: [f64; 3], prefilter: bool) -> Array<A, Ix3>
where
    S: Data<Elem = A>,
    A: Copy + Num + FromPrimitive + ToPrimitive,
{
    let mut o_dim = data.raw_dim();
    for (ax, (&ax_len, zoom)) in data.shape().iter().zip(zoom.iter()).enumerate() {
        o_dim[ax] = (ax_len as f64 * zoom).round() as usize;
    }

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

    let mut out = array_like(&data, o_dim, A::zero());

    println!("Starting reslicer with");
    println!("idim: {:?}", data.dim());
    println!("odim: {:?}", o_dim);
    println!("zoom: {:?}", zoom);
    let reslicer = ZoomShiftReslicer::new(
        [data.dim().0, data.dim().1, data.dim().2],
        [o_dim[0], o_dim[1], o_dim[2]],
        zoom,
        [0.0, 0.0, 0.0],
    );

    let order = 3;
    if prefilter && order > 1 {
        let data = spline_filter(data, 3);
        Zip::indexed(&mut out).for_each(|idx, o| {
            *o = A::from_f64(reslicer.interpolate(&data, idx)).unwrap();
        });
    } else {
        Zip::indexed(&mut out).for_each(|idx, o| {
            *o = A::from_f64(reslicer.interpolate(&data, idx)).unwrap();
        });
    }
    out
}

/// Zoom shift transformation (only scaling and translation).
///
/// Hardcoded to use spline interpolation of order 3 and mirror mode.
struct ZoomShiftReslicer {
    offsets: [Vec<isize>; 3],
    edge_offsets: [Array2<isize>; 3],
    is_edge_case: [Vec<bool>; 3],
    splvals: [Array2<f64>; 3],
}

impl ZoomShiftReslicer {
    /// Build all necessary data to call `interpolate`.
    pub fn new(
        idim: [usize; 3],
        odim: [usize; 3],
        zooms: [f64; 3],
        shifts: [f64; 3],
    ) -> ZoomShiftReslicer {
        let order = 3usize;

        let mut offsets = [vec![0; odim[0]], vec![0; odim[1]], vec![0; odim[2]]];
        let mut edge_offsets = [
            Array2::zeros((odim[0], order + 1)),
            Array2::zeros((odim[1], order + 1)),
            Array2::zeros((odim[2], order + 1)),
        ];
        let mut is_edge_case = [vec![false; odim[0]], vec![false; odim[1]], vec![false; odim[2]]];
        let mut splvals = [
            Array2::zeros((odim[0], order + 1)),
            Array2::zeros((odim[1], order + 1)),
            Array2::zeros((odim[2], order + 1)),
        ];

        let idim = [idim[0] as isize, idim[1] as isize, idim[2] as isize];
        let odim = [odim[0], odim[1], odim[2]];

        for axis in 0..3 {
            let zoom = zooms[axis];
            let shift = shifts[axis];
            let offsets = &mut offsets[axis];
            let edge_offsets = &mut edge_offsets[axis];
            let is_edge_case = &mut is_edge_case[axis];
            let splvals = &mut splvals[axis];
            for from in 0..odim[axis] {
                let to = (from as f64 + shift) * zoom;
                let start = to.floor() as isize - 1;
                offsets[from] = start;

                if start < 0 || start + 3 >= idim[axis] {
                    is_edge_case[from] = true;
                    for o in 0..=order {
                        let mut idx = start + o as isize;
                        let len = idim[axis];
                        let s2 = 2 * len - 2;
                        if idx < 0 {
                            idx = s2 * (-idx / s2) + idx;
                            idx = if idx <= 1 - len { idx + s2 } else { -idx };
                        } else {
                            idx -= s2 * (idx / s2);
                            if idx >= len {
                                idx = s2 - idx;
                            }
                        }
                        edge_offsets[(from, o)] = idx - start;
                    }
                }

                let x = to - to.floor();
                let y = x;
                let z = 1.0 - x;
                splvals[(from, 0)] = z * z * z / 6.0;
                splvals[(from, 1)] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
                splvals[(from, 2)] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
                splvals[(from, order)] =
                    1.0 - (splvals[(from, 0)] + splvals[(from, 1)] + splvals[(from, 2)]);
            }
        }

        ZoomShiftReslicer { offsets, edge_offsets, is_edge_case, splvals }
    }

    /// Spline interpolation with up-to 8 neighbors of a point.
    pub fn interpolate<A, S>(&self, data: &ArrayBase<S, Ix3>, start: (usize, usize, usize)) -> f64
    where
        S: Data<Elem = A>,
        A: ToPrimitive + Add<Output = A> + Sub<Output = A> + Copy,
    {
        // Linear interpolation use a 4x4x4 block. This is simple enough, but we must adjust this
        // block when the `start` is near the edges.
        let valid_index = |original_offset, is_edge, start, d: usize, v| {
            (original_offset + if is_edge { self.edge_offsets[d][(start, v)] } else { v as isize })
                as usize
        };

        let original_offset = self.offsets[0][start.0];
        let is_edge = self.is_edge_case[0][start.0];
        let xs = [
            valid_index(original_offset, is_edge, start.0, 0, 0),
            valid_index(original_offset, is_edge, start.0, 0, 1),
            valid_index(original_offset, is_edge, start.0, 0, 2),
            valid_index(original_offset, is_edge, start.0, 0, 3),
        ];

        let original_offset = self.offsets[1][start.1];
        let is_edge = self.is_edge_case[1][start.1];
        let ys = [
            valid_index(original_offset, is_edge, start.1, 1, 0),
            valid_index(original_offset, is_edge, start.1, 1, 1),
            valid_index(original_offset, is_edge, start.1, 1, 2),
            valid_index(original_offset, is_edge, start.1, 1, 3),
        ];

        let is_edge = self.is_edge_case[2][start.2];
        let original_offset = self.offsets[2][start.2];
        let zs = [
            valid_index(original_offset, is_edge, start.2, 2, 0),
            valid_index(original_offset, is_edge, start.2, 2, 1),
            valid_index(original_offset, is_edge, start.2, 2, 2),
            valid_index(original_offset, is_edge, start.2, 2, 3),
        ];

        let mut t = 0.0;
        for (z, &idx_z) in zs.iter().enumerate() {
            let spline_z = self.splvals[2][(start.2, z)];
            for (y, &idx_y) in ys.iter().enumerate() {
                let spline_yz = self.splvals[1][(start.1, y)] * spline_z;
                for (x, &idx_x) in xs.iter().enumerate() {
                    let spline_xyz = self.splvals[0][(start.0, x)] * spline_yz;
                    t += data[(idx_x, idx_y, idx_z)].to_f64().unwrap() * spline_xyz;
                }
            }
        }
        t
    }
}