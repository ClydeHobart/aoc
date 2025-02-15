use {
    super::CellIter2D,
    bitvec::prelude::*,
    glam::IVec2,
    std::{
        cmp::Ordering,
        collections::{BinaryHeap, HashSet, VecDeque},
        hash::Hash,
        iter::from_fn,
        mem::take,
        ops::{Add, BitAndAssign, Deref, DerefMut},
    },
};

pub struct DepthFirstSearchState<Vertex> {
    stack: Vec<Vertex>,
    explored: HashSet<Vertex>,
    neighbors: Vec<Vertex>,
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

pub trait DepthFirstSearch: Sized {
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

pub struct BreadthFirstSearchState<Vertex> {
    queue: VecDeque<Vertex>,
    explored: HashSet<Vertex>,
    neighbors: Vec<Vertex>,
}

impl<Vertex> BreadthFirstSearchState<Vertex> {
    fn clear(&mut self) {
        self.queue.clear();
        self.explored.clear();
        self.neighbors.clear();
    }
}

impl<Vertex> Default for BreadthFirstSearchState<Vertex> {
    fn default() -> Self {
        Self {
            queue: Default::default(),
            explored: Default::default(),
            neighbors: Default::default(),
        }
    }
}

/// An implementation of https://en.wikipedia.org/wiki/Breadth-first_search
pub trait BreadthFirstSearch: Sized {
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

pub struct OpenSetElement<V: Clone + PartialEq, C: Clone + Ord>(pub V, pub C);

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

/// An implementation of https://en.wikipedia.org/wiki/A*_search_algorithm
pub trait AStar: Sized {
    type Vertex: Clone + Eq + Hash;
    type Cost: Add<Self::Cost, Output = Self::Cost> + Clone + Ord + Sized;

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
    fn update_vertex(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        heuristic: Self::Cost,
    );
    fn reset(&mut self);

    fn run(&mut self) -> Option<Vec<Self::Vertex>> {
        self.reset();

        let start: Self::Vertex = self.start().clone();

        let mut open_set_heap: BinaryHeap<OpenSetElement<Self::Vertex, Self::Cost>> =
            BinaryHeap::new();
        let mut open_set_set: HashSet<Self::Vertex> = HashSet::new();

        open_set_heap.push(OpenSetElement(
            start.clone(),
            self.cost_from_start(&start) + self.heuristic(&start),
        ));
        open_set_set.insert(start);

        let mut neighbors: Vec<OpenSetElement<Self::Vertex, Self::Cost>> = Vec::new();
        let mut neighbor_updates: Vec<NeighborUpdate> = Vec::new();

        while let Some(OpenSetElement(current, _)) = open_set_heap.pop() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            let start_to_current: Self::Cost = self.cost_from_start(&current);

            open_set_set.remove(&current);
            self.neighbors(&current, &mut neighbors);

            if !neighbors.is_empty() {
                neighbor_updates.reserve(neighbors.len());

                let mut neighbor_updates_count: usize = 0_usize;
                let mut any_update_was_in_open_set_set: bool = false;

                for OpenSetElement(neighbor, cost) in neighbors.iter_mut() {
                    let start_to_neighbor: Self::Cost = start_to_current.clone() + cost.clone();

                    if start_to_neighbor < self.cost_from_start(neighbor) {
                        let neighbor_heuristic: Self::Cost = self.heuristic(neighbor);

                        self.update_vertex(
                            &current,
                            neighbor,
                            start_to_neighbor.clone(),
                            neighbor_heuristic.clone(),
                        );

                        let is_in_open_set: bool = !open_set_set.insert(neighbor.clone());

                        *cost = start_to_neighbor + neighbor_heuristic;
                        neighbor_updates.push(NeighborUpdate {
                            cost_is_lower: true,
                            is_in_open_set,
                        });
                        neighbor_updates_count += 1_usize;
                        any_update_was_in_open_set_set |= is_in_open_set;
                    } else {
                        neighbor_updates.push(NeighborUpdate {
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
                        take(&mut open_set_heap).into_vec();

                    let old_element_count: usize = open_set_elements.len();

                    for (open_set_element, neighbor_update) in
                        neighbors.drain(..).zip(neighbor_updates.drain(..))
                    {
                        if neighbor_update.cost_is_lower {
                            if neighbor_update.is_in_open_set {
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

                    open_set_heap = open_set_elements.into();
                } else if neighbor_updates_count > 0_usize {
                    // None of the neighbors were previously in the open set, so just add all
                    // normally
                    open_set_heap.extend(
                        neighbors
                            .drain(..)
                            .zip(neighbor_updates.drain(..))
                            .filter_map(|(open_set_element, neighbor_update)| {
                                if neighbor_update.cost_is_lower {
                                    Some(open_set_element)
                                } else {
                                    None
                                }
                            }),
                    );
                }

                neighbors.clear();
                neighbor_updates.clear();
            }
        }

        None
    }
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

struct NeighborUpdate {
    cost_is_lower: bool,
    is_in_open_set: bool,
}

pub trait Dijkstra: Sized {
    type Vertex: Clone + Eq + Hash;
    type Cost: Add<Self::Cost, Output = Self::Cost> + Clone + Ord + Sized;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn cost_from_start(&self, vertex: &Self::Vertex) -> Self::Cost;
    fn neighbors(
        &self,
        vertex: &Self::Vertex,
        neighbors: &mut Vec<OpenSetElement<Self::Vertex, Self::Cost>>,
    );
    fn update_vertex(&mut self, from: &Self::Vertex, to: &Self::Vertex, cost: Self::Cost);
    fn reset(&mut self);

    fn run(&mut self) -> Option<Vec<Self::Vertex>> {
        self.reset();

        let start: Self::Vertex = self.start().clone();

        let mut open_set_heap: BinaryHeap<OpenSetElement<Self::Vertex, Self::Cost>> =
            BinaryHeap::new();
        let mut open_set_set: HashSet<Self::Vertex> = HashSet::new();

        open_set_heap.push(OpenSetElement(start.clone(), self.cost_from_start(&start)));
        open_set_set.insert(start);

        let mut neighbors: Vec<OpenSetElement<Self::Vertex, Self::Cost>> = Vec::new();
        let mut neighbor_updates: Vec<NeighborUpdate> = Vec::new();

        while let Some(OpenSetElement(current, start_to_current)) = open_set_heap.pop() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            open_set_set.remove(&current);
            self.neighbors(&current, &mut neighbors);

            if !neighbors.is_empty() {
                neighbor_updates.reserve(neighbors.len());

                let mut neighbor_updates_count: usize = 0_usize;
                let mut any_update_was_in_open_set_set: bool = false;

                for OpenSetElement(neighbor, neighbor_cost) in neighbors.iter_mut() {
                    let start_to_neighbor: Self::Cost =
                        start_to_current.clone() + neighbor_cost.clone();

                    if start_to_neighbor < self.cost_from_start(neighbor) {
                        self.update_vertex(&current, neighbor, start_to_neighbor.clone());

                        let is_in_open_set: bool = !open_set_set.insert(neighbor.clone());

                        *neighbor_cost = start_to_neighbor;
                        neighbor_updates.push(NeighborUpdate {
                            cost_is_lower: true,
                            is_in_open_set,
                        });
                        neighbor_updates_count += 1_usize;
                        any_update_was_in_open_set_set |= is_in_open_set;
                    } else {
                        neighbor_updates.push(NeighborUpdate {
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
                        take(&mut open_set_heap).into_vec();

                    let old_element_count: usize = open_set_elements.len();

                    for (
                        open_set_element,
                        NeighborUpdate {
                            cost_is_lower,
                            is_in_open_set,
                        },
                    ) in neighbors.drain(..).zip(neighbor_updates.drain(..))
                    {
                        if cost_is_lower {
                            if is_in_open_set {
                                if let Some(OpenSetElement(_, start_to_neighbor)) =
                                    open_set_elements[..old_element_count].iter_mut().find(
                                        |OpenSetElement(vertex, _)| *vertex == open_set_element.0,
                                    )
                                {
                                    *start_to_neighbor = open_set_element.1;
                                }
                            } else {
                                open_set_elements.push(open_set_element);
                            }
                        }
                    }

                    open_set_heap = open_set_elements.into();
                } else if neighbor_updates_count > 0_usize {
                    open_set_heap.extend(
                        neighbors
                            .drain(..)
                            .zip(neighbor_updates.drain(..))
                            .filter_map(|(open_set_element, neighbor_update)| {
                                if neighbor_update.cost_is_lower {
                                    Some(open_set_element)
                                } else {
                                    None
                                }
                            }),
                    );
                }

                neighbors.clear();
                neighbor_updates.clear();
            }
        }

        None
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

pub trait AdditionalCliqueStateTrait: Clone + Default {
    /// Returns whether or not this is still a valid clique.
    fn is_valid(&self) -> bool;

    /// Returns `Greater` if `self` is more "maximal", etc.
    fn maximal_cmp(&self, other: &Self) -> Ordering;

    /// Returns whether `maximal_cmp` always returns `Equal`.
    fn maximal_cmp_always_returns_equal(&self) -> bool;
}

#[cfg_attr(test, derive(Debug, PartialEq))]
#[derive(Default)]
pub struct CliqueState<BA, A> {
    pub clique: BA,
    pub clique_cardinality: usize,
    pub additional_clique_state: A,
}

#[derive(Default)]
struct StackState<BA, A> {
    neighbors: BA,
    vertex_index: usize,
    additional_clique_state: A,
}

impl<BS: BitStore, BO: BitOrder, BA: Deref<Target = BitSlice<BS, BO>>, A> StackState<BA, A> {
    fn next_neighbor_vertex_index(&self) -> usize {
        self.vertex_index + self.neighbors[self.vertex_index..].leading_zeros()
    }

    fn neighbor_cardinality(&self) -> usize {
        self.neighbors[self.vertex_index..].count_ones()
    }
}

pub trait MaximumClique<BS = usize, BO = Lsb0>
where
    BS: BitStore,
    BO: BitOrder,
{
    type BitArray: for<'b> BitAndAssign<&'b Self::BitArray>
        + Clone
        + Default
        + Deref<Target = BitSlice<BS, BO>>
        + DerefMut;
    type AdditionalCliqueState: AdditionalCliqueStateTrait;

    fn vertex_count(&self) -> usize;
    fn integrate_vertex(
        &self,
        additional_clique_state: &mut Self::AdditionalCliqueState,
        vertex_index: usize,
    );
    fn get_neighbors(&self, vertex_index: usize) -> &Self::BitArray;
    fn run(&self) -> CliqueState<Self::BitArray, Self::AdditionalCliqueState> {
        let vertex_count: usize = self.vertex_count();
        let mut maximal_clique_state: CliqueState<Self::BitArray, Self::AdditionalCliqueState> =
            CliqueState::default();
        let mut clique: Self::BitArray = Default::default();
        let mut clique_cardinality: usize = 0_usize;
        let mut neighbors: Self::BitArray = Default::default();

        neighbors[..vertex_count].fill(true);

        let mut stack: Vec<StackState<Self::BitArray, Self::AdditionalCliqueState>> =
            vec![StackState {
                neighbors,
                ..Default::default()
            }];

        while let Some(stack_state) = stack.last_mut() {
            let vertex_index: usize = stack_state.vertex_index;

            if vertex_index >= vertex_count {
                stack.pop();
            } else if clique[vertex_index] {
                clique.set(vertex_index, false);
                clique_cardinality -= 1_usize;
                stack_state.vertex_index += 1_usize;
                stack_state.vertex_index = stack_state.next_neighbor_vertex_index();
            } else {
                let next_neighbor_vertex_index: usize = stack_state.next_neighbor_vertex_index();

                if next_neighbor_vertex_index > vertex_index {
                    stack_state.vertex_index = next_neighbor_vertex_index;
                } else {
                    clique.set(vertex_index, true);
                    clique_cardinality += 1_usize;

                    let mut additional_clique_state: Self::AdditionalCliqueState =
                        stack_state.additional_clique_state.clone();

                    self.integrate_vertex(&mut additional_clique_state, vertex_index);

                    if additional_clique_state.is_valid() {
                        if clique_cardinality
                            .cmp(&maximal_clique_state.clique_cardinality)
                            .then_with(|| {
                                additional_clique_state
                                    .maximal_cmp(&maximal_clique_state.additional_clique_state)
                            })
                            .is_gt()
                        {
                            maximal_clique_state = CliqueState {
                                clique: clique.clone(),
                                clique_cardinality,
                                additional_clique_state: additional_clique_state.clone(),
                            };
                        }

                        let mut next_stack_state: StackState<
                            Self::BitArray,
                            Self::AdditionalCliqueState,
                        > = StackState {
                            neighbors: stack_state.neighbors.clone(),
                            vertex_index: vertex_index + 1_usize,
                            additional_clique_state,
                        };

                        next_stack_state
                            .neighbors
                            .bitand_assign(self.get_neighbors(vertex_index));
                        next_stack_state.vertex_index =
                            next_stack_state.next_neighbor_vertex_index();

                        if (clique_cardinality + next_stack_state.neighbor_cardinality())
                            .cmp(&maximal_clique_state.clique_cardinality)
                            .then_with(|| {
                                if next_stack_state
                                    .additional_clique_state
                                    .maximal_cmp_always_returns_equal()
                                {
                                    Ordering::Equal
                                } else {
                                    Ordering::Greater
                                }
                            })
                            .is_gt()
                        {
                            stack.push(next_stack_state);
                        }
                    }
                }
            }
        }

        maximal_clique_state
    }
}
