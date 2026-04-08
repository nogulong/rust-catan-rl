use super::{Coord, CoordType, DetailedCoordType};
use crate::board::Error;

pub type TopologyResult = Result<Vec<Coord>, Error>;

pub trait RawTopology {
    fn neighbours(&self, coord: Coord, center_type: CoordType, neighbour_type: CoordType) -> TopologyResult;
}

pub trait Topology {
    fn hex_hex_neighbours(&self, coord: Coord) -> TopologyResult;
    fn hex_path_neighbours(&self, coord: Coord) -> TopologyResult;
    fn hex_intersection_neighbours(&self, coord: Coord) -> TopologyResult;

    fn path_hex_neighbours(&self, coord: Coord) -> TopologyResult;
    fn path_path_neighbours(&self, coord: Coord) -> TopologyResult;
    fn path_intersection_neighbours(&self, coord: Coord) -> TopologyResult;

    fn intersection_hex_neighbours(&self, coord: Coord) -> TopologyResult;
    fn intersection_path_neighbours(&self, coord: Coord) -> TopologyResult;
    fn intersection_intersection_neighbours<'a>(&self, coord: Coord) -> TopologyResult;
}

pub struct CoordTopology;

fn c(x: i8, y: i8) -> Coord {
    Coord::new(x,y)
}

impl RawTopology for CoordTopology {
    fn neighbours(&self, coord :Coord, center_type: CoordType, neighbour_type: CoordType) -> TopologyResult {
        if coord.get_type() != center_type {
            return Err(Error::WrongCoordType { expected: center_type,  received: coord.get_type() });
        }
        let x = coord.x;
        let y = coord.y;
        Ok(match (coord.get_detailed_type(), neighbour_type) {
            (DetailedCoordType::OHex, CoordType::Hex) => vec![c(x+4,y), c(x+2,y+2), c(x-2,y+2), c(x-4,y), c(x-2,y-2), c(x+2,y-2)],
            (DetailedCoordType::OHex, CoordType::Path) => vec![c(x+2,y), c(x+1,y+1), c(x-1,y+1), c(x-2,y), c(x-1,y-1), c(x+1,y-1)],
            (DetailedCoordType::OHex, CoordType::Intersection) => vec![c(x+2,y+1), c(x,y+1), c(x-2,y+1), c(x-2,y-1), c(x,y-1), c(x+2,y-1)],

            (DetailedCoordType::IPath, CoordType::Hex) => vec![c(x+2,y), c(x-2,y)],
            (DetailedCoordType::SPath, CoordType::Hex) => vec![c(x+1,y+1), c(x-1,y-1)],
            (DetailedCoordType::ZPath, CoordType::Hex) => vec![c(x-1,y+1), c(x+1,y-1)],
            (DetailedCoordType::IPath, CoordType::Path) => vec![c(x+1,y+1), c(x-1,y+1), c(x-1,y-1), c(x+1,y-1)],
            (DetailedCoordType::SPath, CoordType::Path) => vec![c(x+2,y), c(x-1,y+1), c(x-2,y), c(x+1,y-1)],
            (DetailedCoordType::ZPath, CoordType::Path) => vec![c(x+2,y), c(x+1,y+1), c(x-2,y), c(x-1,y-1)],
            (DetailedCoordType::IPath, CoordType::Intersection) => vec![c(x,y+1), c(x,y-1)],
            (DetailedCoordType::SPath, CoordType::Intersection) |
            (DetailedCoordType::ZPath, CoordType::Intersection) => vec![c(x+1,y), c(x-1,y)],

            (DetailedCoordType::AIntersection, CoordType::Hex) => vec![c(x+2,y+1), c(x-2,y+1), c(x,y-1)],
            (DetailedCoordType::VIntersection, CoordType::Hex) => vec![c(x,y+1), c(x-2,y-1), c(x+2,y-1)],
            (DetailedCoordType::AIntersection, CoordType::Path) => vec![c(x+1,y), c(x,y+1), c(x-1,y)],
            (DetailedCoordType::VIntersection, CoordType::Path) => vec![c(x+1,y), c(x-1,y), c(x,y-1)],
            (DetailedCoordType::AIntersection, CoordType::Intersection) => vec![c(x+2,y), c(x,y+2), c(x-2,y)],
            (DetailedCoordType::VIntersection, CoordType::Intersection) => vec![c(x+2,y), c(x-2,y), c(x,y-2)],

            _ => return Err(Error::InvalidNeighbourTypes { center:center_type , neighbours:neighbour_type }),
        })
    }
}

impl<T : RawTopology> Topology for T {
    fn hex_hex_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Hex, CoordType::Hex)
    }
    fn hex_path_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Hex, CoordType::Path)
    }
    fn hex_intersection_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Hex, CoordType::Intersection)
    }

    fn path_hex_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Path, CoordType::Hex)
    }
    fn path_path_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Path, CoordType::Path)
    }
    fn path_intersection_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Path, CoordType::Intersection)
    }

    fn intersection_hex_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Intersection, CoordType::Hex)
    }
    fn intersection_path_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Intersection, CoordType::Path)
    }
    fn intersection_intersection_neighbours(&self, coord: Coord) -> TopologyResult {
        self.neighbours(coord, CoordType::Intersection, CoordType::Intersection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_hex_neighbours() {
        let hex = Coord::new(0, 0);
        let topology = CoordTopology;
        let neighbours = topology.hex_hex_neighbours(hex).unwrap();

        assert_eq!(neighbours.len(), 6);
        assert!(neighbours.contains(&Coord::new(4, 0)));
        assert!(neighbours.contains(&Coord::new(2, 2)));
        assert!(neighbours.contains(&Coord::new(-2, 2)));
        assert!(neighbours.contains(&Coord::new(-4, 0)));
        assert!(neighbours.contains(&Coord::new(-2, -2)));
        assert!(neighbours.contains(&Coord::new(2, -2)));
    }

    #[test]
    fn test_hex_path_neighbours() {
        let hex = Coord::new(0, 0);
        let topology = CoordTopology;
        let neighbours = topology.hex_path_neighbours(hex).unwrap();

        assert_eq!(neighbours.len(), 6);
        assert!(neighbours.contains(&Coord::new(2, 0)));
        assert!(neighbours.contains(&Coord::new(1, 1)));
        assert!(neighbours.contains(&Coord::new(-1, 1)));
        assert!(neighbours.contains(&Coord::new(-2, 0)));
        assert!(neighbours.contains(&Coord::new(-1, -1)));
        assert!(neighbours.contains(&Coord::new(1, -1)));
    }

    #[test]
    fn test_hex_intersection_neighbours() {
        let hex = Coord::new(0, 0);
        let topology = CoordTopology;
        let neighbours = topology.hex_intersection_neighbours(hex).unwrap();

        assert_eq!(neighbours.len(), 6);
        assert!(neighbours.contains(&Coord::new(2, 1)));
        assert!(neighbours.contains(&Coord::new(0, 1)));
        assert!(neighbours.contains(&Coord::new(-2, 1)));
        assert!(neighbours.contains(&Coord::new(-2, -1)));
        assert!(neighbours.contains(&Coord::new(0, -1)));
        assert!(neighbours.contains(&Coord::new(2, -1)));
    }

    #[test]
    fn test_path_hex_neighbours() {
        let path = Coord::new(2, 0);  // IPath
        let topology = CoordTopology;
        let neighbours = topology.path_hex_neighbours(path).unwrap();

        assert_eq!(neighbours.len(), 2);
        assert!(neighbours.contains(&Coord::new(4, 0)));
        assert!(neighbours.contains(&Coord::new(0, 0)));
    }

    #[test]
    fn test_path_intersection_neighbours() {
        let path = Coord::new(2, 0);  // IPath
        let topology = CoordTopology;
        let neighbours = topology.path_intersection_neighbours(path).unwrap();

        assert_eq!(neighbours.len(), 2);
        assert!(neighbours.contains(&Coord::new(2, 1)));
        assert!(neighbours.contains(&Coord::new(2, -1)));
    }

    #[test]
    fn test_intersection_hex_neighbours() {
        let intersection = Coord::new(0, 1);  // AIntersection
        let topology = CoordTopology;
        let neighbours = topology.intersection_hex_neighbours(intersection).unwrap();

        assert_eq!(neighbours.len(), 3);
        assert!(neighbours.contains(&Coord::new(2, 2)));
        assert!(neighbours.contains(&Coord::new(-2, 2)));
        assert!(neighbours.contains(&Coord::new(0, 0)));
    }

    #[test]
    fn test_intersection_path_neighbours() {
        let intersection = Coord::new(0, 1);  // AIntersection
        let topology = CoordTopology;
        let neighbours = topology.intersection_path_neighbours(intersection).unwrap();

        assert_eq!(neighbours.len(), 3);
        assert!(neighbours.contains(&Coord::new(1, 1)));
        assert!(neighbours.contains(&Coord::new(0, 2)));
        assert!(neighbours.contains(&Coord::new(-1, 1)));
    }

    #[test]
    fn test_intersection_intersection_neighbours() {
        let intersection = Coord::new(0, 1);  // AIntersection
        let topology = CoordTopology;
        let neighbours = topology.intersection_intersection_neighbours(intersection).unwrap();

        assert_eq!(neighbours.len(), 3);
        assert!(neighbours.contains(&Coord::new(2, 1)));
        assert!(neighbours.contains(&Coord::new(0, 3)));
        assert!(neighbours.contains(&Coord::new(-2, 1)));
    }

    #[test]
    fn test_wrong_coord_type_error() {
        let path = Coord::new(2, 0);  // This is a Path
        let topology = CoordTopology;

        // Try to get hex_hex_neighbours on a path - should fail
        let result = topology.hex_hex_neighbours(path);
        assert!(result.is_err());

        if let Err(Error::WrongCoordType { expected, received }) = result {
            assert_eq!(expected, CoordType::Hex);
            assert_eq!(received, CoordType::Path);
        } else {
            panic!("Expected WrongCoordType error");
        }
    }
}