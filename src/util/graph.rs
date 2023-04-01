use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet, VecDeque},
    fmt::Debug,
    hash::Hash,
    mem::take,
    ops::Add,
};

struct OpenSetElement<V: Clone + PartialEq, C: Clone + Ord>(V, C);

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
    fn cost_between_neighbors(&self, from: &Self::Vertex, to: &Self::Vertex) -> Self::Cost;
    fn heuristic(&self, vertex: &Self::Vertex) -> Self::Cost;
    fn neighbors(&self, vertex: &Self::Vertex, neighbors: &mut Vec<Self::Vertex>);
    fn update_score(
        &mut self,
        from: &Self::Vertex,
        to: &Self::Vertex,
        cost: Self::Cost,
        heuristic: Self::Cost,
    );

    fn run(mut self) -> Option<Vec<Self::Vertex>> {
        let start: Self::Vertex = self.start().clone();

        let mut open_set_heap: BinaryHeap<OpenSetElement<Self::Vertex, Self::Cost>> =
            BinaryHeap::new();
        let mut open_set_set: HashSet<Self::Vertex> = HashSet::new();

        open_set_heap.push(OpenSetElement(start.clone(), self.cost_from_start(&start)));
        open_set_set.insert(start);

        let mut neighbors: Vec<Self::Vertex> = Vec::new();

        // A pair, where the first field is the new cost for the neighbor, already passed into
        // `update_score`, and a bool representing whether the neighbor was previously in
        // `open_set_set`, meaning `open_set_heap` requires special attention to update its score
        let mut neighbor_updates: Vec<Option<(Self::Cost, bool)>> = Vec::new();
        let mut any_update_was_in_open_set_set: bool = false;

        while let Some(OpenSetElement(current, _)) = open_set_heap.pop() {
            if self.is_end(&current) {
                return Some(self.path_to(&current));
            }

            let start_to_current: Self::Cost = self.cost_from_start(&current);

            open_set_set.remove(&current);
            self.neighbors(&current, &mut neighbors);
            neighbor_updates.reserve(neighbors.len());

            for neighbor in neighbors.iter() {
                let start_to_neighbor: Self::Cost =
                    start_to_current.clone() + self.cost_between_neighbors(&current, &neighbor);

                if start_to_neighbor < self.cost_from_start(&neighbor) {
                    let neighbor_heuristic: Self::Cost = self.heuristic(&neighbor);

                    self.update_score(
                        &current,
                        &neighbor,
                        start_to_neighbor.clone(),
                        neighbor_heuristic.clone(),
                    );

                    let was_in_open_set_set: bool = !open_set_set.insert(neighbor.clone());

                    neighbor_updates.push(Some((
                        start_to_neighbor + neighbor_heuristic,
                        was_in_open_set_set,
                    )));
                    any_update_was_in_open_set_set |= was_in_open_set_set;
                } else {
                    neighbor_updates.push(None);
                }
            }

            if any_update_was_in_open_set_set {
                // Convert to a vec first, add the new elements, then convert back, so that we don't
                // waste time during `push` operations only to have that effort ignored when
                // converting back to a heap
                let mut open_set_elements: Vec<OpenSetElement<Self::Vertex, Self::Cost>> =
                    take(&mut open_set_heap).into_vec();

                let old_element_count: usize = open_set_elements.len();

                // None of the neighbors were previously in the open set, so just add all normally
                for (neighbor, neighbor_update) in neighbors.iter().zip(neighbor_updates.iter()) {
                    if let Some((cost, is_in_open_set_elements)) = neighbor_update {
                        if *is_in_open_set_elements {
                            if let Some(index) = open_set_elements[..old_element_count]
                                .iter()
                                .position(|OpenSetElement(vertex, _)| *vertex == *neighbor)
                            {
                                open_set_elements[index].1 = cost.clone();
                            }
                        } else {
                            open_set_elements.push(OpenSetElement(neighbor.clone(), cost.clone()));
                        }
                    }
                }

                open_set_heap = open_set_elements.into();
            } else {
                // None of the neighbors were previously in the open set, so just add all normally
                for (neighbor, neighbor_update) in neighbors.iter().zip(neighbor_updates.iter()) {
                    if let Some((cost, _)) = neighbor_update {
                        open_set_heap.push(OpenSetElement(neighbor.clone(), cost.clone()));
                    }
                }
            }

            neighbors.clear();
            neighbor_updates.clear();
            any_update_was_in_open_set_set = false;
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

    fn run(mut self) -> Option<Vec<Self::Vertex>> {
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
