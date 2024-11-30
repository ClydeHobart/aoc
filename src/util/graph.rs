use {
    super::CellIter2D,
    glam::IVec2,
    std::{
        cmp::Ordering,
        collections::{BinaryHeap, HashSet, VecDeque},
        fmt::Debug,
        hash::Hash,
        iter::from_fn,
        mem::take,
        ops::Add,
    },
};

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

/// An implementation of https://en.wikipedia.org/wiki/Breadth-first_search
pub trait BreadthFirstSearch: Sized {
    type Vertex: Clone + Debug + Eq + Hash;

    fn start(&self) -> &Self::Vertex;
    fn is_end(&self, vertex: &Self::Vertex) -> bool;
    fn path_to(&self, vertex: &Self::Vertex) -> Vec<Self::Vertex>;
    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn update_parent(&mut self, from: &Self::Vertex, to: &Self::Vertex);
    fn reset(&mut self);

    fn run(&mut self) -> Option<Vec<Self::Vertex>> {
        self.reset();

        let mut queue: VecDeque<Self::Vertex> = VecDeque::new();
        let mut explored: HashSet<Self::Vertex> = HashSet::new();

        let start: Self::Vertex = self.start().clone();
        explored.insert(start.clone());
        queue.push_back(start);

        let mut neighbors: Vec<Self::Vertex> = Vec::new();

        while let Some(current) = queue.pop_front() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            self.neighbors(&current, &mut neighbors);

            for neighbor in neighbors.drain(..) {
                if explored.insert(neighbor.clone()) {
                    self.update_parent(&current, &neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }
}

/// An implementation of [Kahn's Algorithm][kahn] for producing a topological sort of a DAG.
///
/// [kahn]: https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
pub trait Kahn {
    type Vertex: Clone + Debug + Eq + Hash;

    fn populate_initial_set(&self, initial_set: &mut Vec<Self::Vertex>);
    fn out_neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn remove_edge(&mut self, from: &Self::Vertex, to: &Self::Vertex);
    fn has_in_neighbors(&self, vertex: &Self::Vertex) -> bool;
    fn any_edges_exist(&self) -> bool;
    fn reset(&mut self);

    fn run(&mut self) -> Option<Vec<Self::Vertex>> {
        let mut list: Vec<Self::Vertex> = Vec::new();
        let mut set: Vec<Self::Vertex> = Vec::new();
        let mut neighbors: Vec<Self::Vertex> = Vec::new();

        self.reset();
        self.populate_initial_set(&mut set);

        while let Some(vertex) = set.pop() {
            list.push(vertex.clone());
            neighbors.clear();
            self.out_neighbors(&vertex, &mut neighbors);

            for neighbor in neighbors.drain(..) {
                self.remove_edge(&vertex, &neighbor);

                if !self.has_in_neighbors(&neighbor) {
                    set.push(neighbor);
                }
            }
        }

        if self.any_edges_exist() {
            None
        } else {
            Some(list)
        }
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
