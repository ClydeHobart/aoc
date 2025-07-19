use {
    super::CellIter2D,
    bitvec::prelude::*,
    glam::IVec2,
    num::Zero,
    std::{
        cmp::Ordering,
        collections::{BinaryHeap, HashSet, VecDeque},
        hash::Hash,
        iter::from_fn,
        mem::take,
        ops::{Add, BitAndAssign, Deref, DerefMut},
    },
};

pub struct DepthFirstSearchState<V> {
    stack: Vec<V>,
    explored: HashSet<V>,
    neighbors: Vec<V>,
}

impl<Vertex> DepthFirstSearchState<Vertex> {
    fn clear(&mut self) {
        self.stack.clear();
        self.explored.clear();
        self.neighbors.clear();
    }
}

impl<Vertex> Default for DepthFirstSearchState<Vertex> {
    fn default() -> Self {
        Self {
            stack: Default::default(),
            explored: Default::default(),
            neighbors: Default::default(),
        }
    }
}

pub trait DepthFirstSearch {
    type Vertex: Clone + Eq + Hash;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex);
    fn reset(&mut self);

    fn run_internal(
        &mut self,
        state: &mut DepthFirstSearchState<Self::Vertex>,
    ) -> Option<Vec<Self::Vertex>> {
        self.reset();

        state.clear();

        let start: Self::Vertex = self.start().clone();
        state.explored.insert(start.clone());
        state.stack.push(start);

        while let Some(current) = state.stack.pop() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            self.neighbors(&current, &mut state.neighbors);

            for neighbor in state.neighbors.drain(..) {
                if state.explored.insert(neighbor.clone()) {
                    self.update_parent(&current, &neighbor);
                    state.stack.push(neighbor);
                }
            }
        }

        None
    }

    fn run(&mut self) -> Option<Vec<Self::Vertex>> {
        self.run_internal(&mut DepthFirstSearchState::default())
    }
}

pub struct BreadthFirstSearchState<V> {
    queue: VecDeque<V>,
    explored: HashSet<V>,
    neighbors: Vec<V>,
}

impl<V> BreadthFirstSearchState<V> {
    fn clear(&mut self) {
        self.queue.clear();
        self.explored.clear();
        self.neighbors.clear();
    }
}

impl<V> Default for BreadthFirstSearchState<V> {
    fn default() -> Self {
        Self {
            queue: Default::default(),
            explored: Default::default(),
            neighbors: Default::default(),
        }
    }
}

/// An implementation of https://en.wikipedia.org/wiki/Breadth-first_search
pub trait BreadthFirstSearch {
    type Vertex: Clone + Eq + Hash;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex);
    fn reset(&mut self);

    fn run_internal(
        &mut self,
        state: &mut BreadthFirstSearchState<Self::Vertex>,
    ) -> Option<Vec<Self::Vertex>> {
        self.reset();

        state.clear();

        let start: Self::Vertex = self.start().clone();
        state.explored.insert(start.clone());
        state.queue.push_back(start);

        while let Some(current) = state.queue.pop_front() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            self.neighbors(&current, &mut state.neighbors);

            for neighbor in state.neighbors.drain(..) {
                if state.explored.insert(neighbor.clone()) {
                    self.update_parent(&current, &neighbor);
                    state.queue.push_back(neighbor);
                }
            }
        }

        None
    }

    fn run(&mut self) -> Option<Vec<Self::Vertex>> {
        self.run_internal(&mut BreadthFirstSearchState::default())
    }
}

pub struct OpenSetElement<V, C>(pub V, pub C);

impl<V: Clone + PartialEq, C: Clone + Ord> PartialEq for OpenSetElement<V, C> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<V: Clone + PartialEq, C: Clone + Ord> PartialOrd for OpenSetElement<V, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse the order so that cost is minimized when popping from the heap
        Some(other.1.cmp(&self.1))
    }
}

impl<V: Clone + PartialEq, C: Clone + Ord> Eq for OpenSetElement<V, C> {}

impl<V: Clone + PartialEq, C: Clone + Ord> Ord for OpenSetElement<V, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the order so that cost is minimized when popping from the heap
        other.1.cmp(&self.1)
    }
}

struct NeighborUpdate {
    cost_is_lower: bool,
    is_in_open_set: bool,
}

pub struct WeightedGraphSearchState<V, C> {
    open_set_heap: BinaryHeap<OpenSetElement<V, C>>,
    open_set_set: HashSet<V>,
    neighbors: Vec<OpenSetElement<V, C>>,
    neighbor_updates: Vec<NeighborUpdate>,
}

impl<V, C> WeightedGraphSearchState<V, C> {
    fn clear(&mut self) {
        self.open_set_heap.clear();
        self.open_set_set.clear();
        self.neighbors.clear();
        self.neighbor_updates.clear();
    }
}

impl<V, C> Default for WeightedGraphSearchState<V, C>
where
    OpenSetElement<V, C>: Ord,
{
    fn default() -> Self {
        Self {
            open_set_heap: Default::default(),
            open_set_set: Default::default(),
            neighbors: Default::default(),
            neighbor_updates: Default::default(),
        }
    }
}

pub fn zero_heuristic<W: WeightedGraphSearch + ?Sized>(
    _search: &W,
    _vertex: &W::Vertex,
) -> W::Cost {
    W::Cost::zero()
}

// fn try_cost_as_u64<C: Clone>(cost: &C) -> Option<u64> {
//     use std::mem::{align_of, size_of, transmute};

//     (size_of::<C>() == size_of::<u64>() && align_of::<C>() <= align_of::<u64>())
//         .then(|| *unsafe { transmute::<&C, &u64>(cost) })
// }

/// An implementation of https://en.wikipedia.org/wiki/A*_search_algorithm and
/// https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm
pub trait WeightedGraphSearch {
    type Vertex: Clone + Eq + Hash;
    type Cost: Add<Self::Cost, Output = Self::Cost> + Clone + Ord + Sized + Zero;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost;
    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost;

    /// The cost is from `vertex` to the neighbor.
    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    );

    /// `heuristic` may be zero if this is called by Dijkstra.
    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        heuristic: Self::Cost,
    );
    fn reset(&mut self);

    fn run_internal<F: Fn(&Self, &Self::Vertex) -> Self::Cost>(
        &mut self,
        state: &mut WeightedGraphSearchState<Self::Vertex, Self::Cost>,
        heuristic: F,
    ) -> Option<Vec<Self::Vertex>> {
        self.reset();
        state.clear();

        let start: Self::Vertex = self.start().clone();

        state.open_set_heap.push(OpenSetElement(
            start.clone(),
            self.cost_from_start(&start) + heuristic(self, &start),
        ));
        state.open_set_set.insert(start);

        while let Some(OpenSetElement(current, _)) = state.open_set_heap.pop() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            let start_to_current: Self::Cost = self.cost_from_start(&current);

            state.open_set_set.remove(&current);
            self.neighbors(&current, &mut state.neighbors);

            if !state.neighbors.is_empty() {
                state.neighbor_updates.reserve(state.neighbors.len());

                let mut neighbor_updates_count: usize = 0_usize;
                let mut any_update_was_in_open_set_set: bool = false;

                for OpenSetElement(neighbor, neighbor_cost) in state.neighbors.iter_mut() {
                    let start_to_neighbor: Self::Cost =
                        start_to_current.clone() + neighbor_cost.clone();

                    if start_to_neighbor < self.cost_from_start(neighbor) {
                        let neighbor_heuristic: Self::Cost = heuristic(self, neighbor);

                        self.update_vertex(
                            &current,
                            neighbor,
                            start_to_neighbor.clone(),
                            neighbor_heuristic.clone(),
                        );

                        let is_in_open_set: bool = !state.open_set_set.insert(neighbor.clone());

                        *neighbor_cost = start_to_neighbor + neighbor_heuristic;
                        state.neighbor_updates.push(NeighborUpdate {
                            cost_is_lower: true,
                            is_in_open_set,
                        });
                        neighbor_updates_count += 1_usize;
                        any_update_was_in_open_set_set |= is_in_open_set;
                    } else {
                        state.neighbor_updates.push(NeighborUpdate {
                            cost_is_lower: false,
                            is_in_open_set: false,
                        });
                    }
                }

                if any_update_was_in_open_set_set {
                    // Convert to a vec first, add the new elements, then convert back, so that we
                    // don't waste time during `push` operations only to have that effort ignored
                    // when converting back to a heap
                    let mut open_set_elements: Vec<OpenSetElement<Self::Vertex, Self::Cost>> =
                        take(&mut state.open_set_heap).into_vec();

                    let old_element_count: usize = open_set_elements.len();

                    for (
                        open_set_element,
                        NeighborUpdate {
                            cost_is_lower,
                            is_in_open_set,
                        },
                    ) in state
                        .neighbors
                        .drain(..)
                        .zip(state.neighbor_updates.drain(..))
                    {
                        if cost_is_lower {
                            if is_in_open_set {
                                if let Some(OpenSetElement(_, cost_mut)) = open_set_elements
                                    [..old_element_count]
                                    .iter_mut()
                                    .find(|OpenSetElement(vertex, _)| *vertex == open_set_element.0)
                                {
                                    *cost_mut = open_set_element.1;
                                }
                            } else {
                                open_set_elements.push(open_set_element);
                            }
                        }
                    }

                    state.open_set_heap = open_set_elements.into();
                } else if neighbor_updates_count > 0_usize {
                    // None of the neighbors were previously in the open set, so just add all
                    // normally
                    state.open_set_heap.extend(
                        state
                            .neighbors
                            .drain(..)
                            .zip(state.neighbor_updates.drain(..))
                            .filter_map(|(open_set_element, neighbor_update)| {
                                if neighbor_update.cost_is_lower {
                                    Some(open_set_element)
                                } else {
                                    None
                                }
                            }),
                    );
                }

                state.neighbors.clear();
                state.neighbor_updates.clear();
            }
        }

        None
    }

    fn run_a_star_internal(
        &mut self,
        state: &mut WeightedGraphSearchState<Self::Vertex, Self::Cost>,
    ) -> Option<Vec<Self::Vertex>> {
        self.run_internal(state, Self::heuristic)
    }

    fn run_a_star(&mut self) -> Option<Vec<Self::Vertex>> {
        self.run_a_star_internal(&mut WeightedGraphSearchState::default())
    }

    fn run_dijkstra_internal(
        &mut self,
        state: &mut WeightedGraphSearchState<Self::Vertex, Self::Cost>,
    ) -> Option<Vec<Self::Vertex>> {
        self.run_internal(state, zero_heuristic::<Self>)
    }

    fn run_dijkstra(&mut self) -> Option<Vec<Self::Vertex>> {
        self.run_dijkstra_internal(&mut WeightedGraphSearchState::default())
    }
}

fn jump_flooding_algoritm_iter_k(n: i32) -> impl Iterator<Item = i32> {
    let mut k: i32 = n;

    from_fn(move || {
        k /= 2_i32;

        (k > 0_i32).then_some(k)
    })
}

/// An implementation of the [Jump Flooding Algorithm][jfa] for constructing a Voronoi diagram.
///
/// [jfa]: https://en.wikipedia.org/wiki/Jump_flooding_algorithm
pub trait JumpFloodingAlgorithm {
    type Pixel: Clone + PartialEq;
    type Dist: PartialOrd;

    fn dist(a: IVec2, b: IVec2) -> Self::Dist;
    fn n(&self) -> i32;
    fn is_pos_valid(&self, pos: IVec2) -> bool;
    fn try_get_p_pixel(&self, pos: IVec2) -> Option<Self::Pixel>;
    fn try_get_q_pixel(&self, pos: IVec2) -> Option<Self::Pixel>;
    fn get_seed(&self, pixel: Self::Pixel) -> IVec2;
    fn update_pixel(&mut self, pos: IVec2, pixel: Self::Pixel, strictly_closer: bool);
    fn reset(&mut self);
    fn on_iteration_complate(&mut self);

    fn run_iteration(&mut self, n: i32, k: i32) {
        let q_array: [i32; 3_usize] = [-k, 0_i32, k];

        for p in CellIter2D::try_from(IVec2::ZERO..IVec2::new(0_i32, n))
            .unwrap()
            .flat_map(|row_pos| CellIter2D::try_from(row_pos..IVec2::new(n, row_pos.y)).unwrap())
        {
            let mut p_pixel_option: Option<Self::Pixel> = self.try_get_p_pixel(p);

            for q in q_array
                .into_iter()
                .flat_map(|j| q_array.into_iter().map(move |i| p + IVec2::new(i, j)))
            {
                if let Some(q_pixel) = (self.is_pos_valid(q) && p != q)
                    .then(|| self.try_get_q_pixel(q))
                    .flatten()
                {
                    if let Some(p_pixel) = p_pixel_option.clone() {
                        if p_pixel != q_pixel {
                            if let Some(ordering) = Self::dist(p, self.get_seed(p_pixel))
                                .partial_cmp(&Self::dist(p, self.get_seed(q_pixel.clone())))
                            {
                                if ordering.is_ge() {
                                    self.update_pixel(p, q_pixel, ordering.is_gt());
                                    p_pixel_option = self.try_get_p_pixel(p);
                                }
                            }
                        }
                    } else {
                        self.update_pixel(p, q_pixel, true);
                        p_pixel_option = self.try_get_p_pixel(p);
                    }
                }
            }
        }

        self.on_iteration_complate();
    }

    fn run_iterations<K: Iterator<Item = i32>>(&mut self, k_iter: K) {
        self.reset();

        let n: i32 = self.n();

        assert!(n >= 0_i32);

        for k in k_iter {
            self.run_iteration(n, k);
        }
    }

    fn run_jfa(&mut self) {
        self.run_iterations(jump_flooding_algoritm_iter_k(self.n()))
    }

    fn run_jfa_plus_one(&mut self) {
        self.run_iterations(jump_flooding_algoritm_iter_k(self.n()).chain([1_i32]))
    }

    fn run_jfa_plus_two(&mut self) {
        self.run_iterations(jump_flooding_algoritm_iter_k(self.n()).chain([2_i32, 1_i32]))
    }

    fn run_one_plus_jfa(&mut self) {
        self.run_iterations(
            [1_i32]
                .into_iter()
                .chain(jump_flooding_algoritm_iter_k(self.n())),
        )
    }

    fn run_jfa_squared(&mut self) {
        self.run_iterations(
            jump_flooding_algoritm_iter_k(self.n()).chain(jump_flooding_algoritm_iter_k(self.n())),
        )
    }
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Clone, Default)]
pub struct CliqueState<BA, UCS> {
    pub clique: BA,
    pub user_clique_state: UCS,
}

#[derive(Default)]
struct StackState<BA, UCS> {
    neighbors: BA,
    vertex_index: usize,
    user_clique_state: UCS,
}

impl<BS: BitStore, BO: BitOrder, BA: Deref<Target = BitSlice<BS, BO>>, UCS> StackState<BA, UCS> {
    fn next_neighbor_vertex_index(&self) -> usize {
        self.vertex_index + self.neighbors[self.vertex_index..].leading_zeros()
    }
}

pub struct CliqueIterator<BA, UCS, UCI, BS = usize, BO = Lsb0>
where
    BA: for<'b> BitAndAssign<&'b BA>
        + Clone
        + Default
        + Deref<Target = BitSlice<BS, BO>>
        + DerefMut,
    UCS: Clone + Default,
    UCI: UserCliqueIteratorTrait<BS, BO, BitArray = BA, UserCliqueState = UCS>,
    BS: BitStore,
    BO: BitOrder,
{
    clique: BA,
    stack: Vec<StackState<BA, UCS>>,
    pub user_clique_iterator: UCI,
}

impl<BA, UCS, UCI, BS, BO> CliqueIterator<BA, UCS, UCI, BS, BO>
where
    BA: for<'b> BitAndAssign<&'b BA>
        + Clone
        + Default
        + Deref<Target = BitSlice<BS, BO>>
        + DerefMut,
    UCS: Clone + Default,
    UCI: UserCliqueIteratorTrait<BS, BO, BitArray = BA, UserCliqueState = UCS>,
    BS: BitStore,
    BO: BitOrder,
{
    pub fn reset(&mut self) {
        let mut neighbors: BA = Default::default();

        neighbors[..self.user_clique_iterator.vertex_count()].fill(true);
        self.stack.clear();
        self.stack.push(StackState {
            neighbors,
            ..Default::default()
        });
    }
}

impl<BA, UCS, UCI, BS, BO> From<UCI> for CliqueIterator<BA, UCS, UCI, BS, BO>
where
    BA: for<'b> BitAndAssign<&'b BA>
        + Clone
        + Default
        + Deref<Target = BitSlice<BS, BO>>
        + DerefMut,
    UCS: Clone + Default,
    UCI: UserCliqueIteratorTrait<BS, BO, BitArray = BA, UserCliqueState = UCS>,
    BS: BitStore,
    BO: BitOrder,
{
    fn from(user_clique_iterator: UCI) -> Self {
        let mut clique_iterator: Self = Self {
            clique: Default::default(),
            stack: Vec::new(),
            user_clique_iterator,
        };

        clique_iterator.reset();

        clique_iterator
    }
}

impl<BA, UCS, UCI, BS, BO> Iterator for CliqueIterator<BA, UCS, UCI, BS, BO>
where
    BA: for<'b> BitAndAssign<&'b BA>
        + Clone
        + Default
        + Deref<Target = BitSlice<BS, BO>>
        + DerefMut,
    UCS: Clone + Default,
    UCI: UserCliqueIteratorTrait<BS, BO, BitArray = BA, UserCliqueState = UCS>,
    BS: BitStore,
    BO: BitOrder,
{
    type Item = CliqueState<BA, UCS>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut next: Option<Self::Item> = None;

        let vertex_count: usize = self.user_clique_iterator.vertex_count();

        while let Some(stack_state) = next.is_none().then(|| self.stack.last_mut()).flatten() {
            let vertex_index: usize = stack_state.vertex_index;

            if vertex_index >= vertex_count {
                // We've visited all vertices for this stack frame.
                self.stack.pop();
            } else if self.clique[vertex_index] {
                // We just visited this vertex, move on to the next one for the stack frame.
                self.clique.set(vertex_index, false);
                stack_state.vertex_index += 1_usize;
                stack_state.vertex_index = stack_state.next_neighbor_vertex_index();
            } else {
                let next_neighbor_vertex_index: usize = stack_state.next_neighbor_vertex_index();

                if next_neighbor_vertex_index > vertex_index {
                    // We're not currently at the correct vertex, just update the vertex index and
                    // do another pass so that we can easily catch the case where there are no more
                    // valid neighbors for this stack frame.
                    stack_state.vertex_index = next_neighbor_vertex_index;
                } else {
                    self.clique.set(vertex_index, true);
                    stack_state.neighbors.set(vertex_index, false);

                    let mut user_clique_state: UCS = stack_state.user_clique_state.clone();

                    self.user_clique_iterator.integrate_vertex(
                        vertex_index,
                        &self.clique,
                        &mut user_clique_state,
                    );

                    if self
                        .user_clique_iterator
                        .is_clique_state_valid(&self.clique, &user_clique_state)
                    {
                        let clique_state: CliqueState<BA, UCS> = CliqueState {
                            clique: self.clique.clone(),
                            user_clique_state: user_clique_state.clone(),
                        };

                        self.user_clique_iterator.visit_clique(&clique_state);

                        next = Some(clique_state);

                        let mut neighbors: BA = Default::default();

                        self.user_clique_iterator.get_neighbors(
                            vertex_index,
                            &self.clique,
                            &user_clique_state,
                            &mut neighbors,
                        );

                        neighbors.bitand_assign(&stack_state.neighbors);

                        if self.user_clique_iterator.should_visit_neighbors(
                            &self.clique,
                            &user_clique_state,
                            &neighbors,
                        ) {
                            let mut next_stack_state: StackState<BA, UCS> = StackState {
                                neighbors,
                                vertex_index: vertex_index + 1_usize,
                                user_clique_state,
                            };

                            next_stack_state.vertex_index =
                                next_stack_state.next_neighbor_vertex_index();

                            self.stack.push(next_stack_state);
                        }
                    }
                }
            }
        }

        next
    }
}

pub trait UserCliqueIteratorTrait<BS = usize, BO = Lsb0>
where
    BS: BitStore,
    BO: BitOrder,
    Self: Sized,
{
    type BitArray: for<'b> BitAndAssign<&'b Self::BitArray>
        + Clone
        + Default
        + Deref<Target = BitSlice<BS, BO>>
        + DerefMut;
    type UserCliqueState: Clone + Default;

    fn iter(self) -> CliqueIterator<Self::BitArray, Self::UserCliqueState, Self, BS, BO> {
        self.into()
    }

    fn vertex_count(&self) -> usize;

    // `vertex_index` is already set in `clique`.
    fn integrate_vertex(
        &self,
        vertex_index: usize,
        clique: &Self::BitArray,
        inout_user_clique_state: &mut Self::UserCliqueState,
    );

    fn is_clique_state_valid(
        &self,
        clique: &Self::BitArray,
        user_clique_state: &Self::UserCliqueState,
    ) -> bool;

    fn visit_clique(&mut self, clique_state: &CliqueState<Self::BitArray, Self::UserCliqueState>);

    // `vertex_index` is already set in `clique`.
    fn get_neighbors(
        &self,
        vertex_index: usize,
        clique: &Self::BitArray,
        user_clique_state: &Self::UserCliqueState,
        out_neighbors: &mut Self::BitArray,
    );

    fn should_visit_neighbors(
        &self,
        clique: &Self::BitArray,
        user_clique_state: &Self::UserCliqueState,
        neighbors: &Self::BitArray,
    ) -> bool;
}

pub struct KahnState<V> {
    pub list: Vec<V>,
    set: VecDeque<V>,
    neighbors: Vec<V>,
}

impl<V> Default for KahnState<V> {
    fn default() -> Self {
        Self {
            list: Default::default(),
            set: Default::default(),
            neighbors: Default::default(),
        }
    }
}

/// An implementation of [Kahn's Algorithm][kahn] for producing a topological sort of a DAG.
///
/// [kahn]: https://en.wikipedia.org/wiki/Topological_sorting#Kahn%27s_algorithm
pub trait Kahn {
    type Vertex: Clone;

    fn populate_initial_set(&self, initial_set: &mut VecDeque<Self::Vertex>);
    fn out_neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn remove_edge(&mut self, from: &Self::Vertex, to: &Self::Vertex);
    fn has_in_neighbors(&self, vertex: &Self::Vertex) -> bool;
    fn any_edges_exist(&self) -> bool;
    fn reset(&mut self);
    fn order_set(&self, set: &mut VecDeque<Self::Vertex>);

    fn run_internal(&mut self, state: &mut KahnState<Self::Vertex>) -> bool {
        state.list.clear();
        state.set.clear();
        state.neighbors.clear();

        self.reset();
        self.populate_initial_set(&mut state.set);

        while let Some(vertex) = state.set.pop_front() {
            state.list.push(vertex.clone());
            state.neighbors.clear();
            self.out_neighbors(&vertex, &mut state.neighbors);

            let mut pushed_into_set: bool = false;

            for neighbor in state.neighbors.drain(..) {
                self.remove_edge(&vertex, &neighbor);

                if !self.has_in_neighbors(&neighbor) {
                    state.set.push_back(neighbor);
                    pushed_into_set = true;
                }
            }

            if pushed_into_set {
                self.order_set(&mut state.set);
            }
        }

        !self.any_edges_exist()
    }

    fn run(&mut self) -> Option<Vec<Self::Vertex>> {
        let mut state: KahnState<Self::Vertex> = KahnState::default();

        self.run_internal(&mut state).then_some(state.list)
    }
}
