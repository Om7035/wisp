use telemetry::DeviceProfile;

#[derive(Debug, Clone, PartialEq)]
pub struct LayerAssignment {
    pub device_id: String,
    pub layer_count: u32,
}

/// Halda-inspired greedy placement: weight by compute_score, cap by how many
/// layers actually fit in each device's available memory, then hand any
/// remainder (from rounding, or from capacity caps) to the fastest device(s)
/// with spare capacity, one layer at a time, fastest first.
pub fn plan_placement(
    devices: &[DeviceProfile],
    total_layers: u32,
    bytes_per_layer: u64,
) -> Vec<LayerAssignment> {
    assert!(!devices.is_empty(), "plan_placement requires at least one device");
    assert!(bytes_per_layer > 0, "bytes_per_layer must be positive");

    let capacities: Vec<u32> = devices
        .iter()
        .map(|d| (d.available_mem_bytes / bytes_per_layer) as u32)
        .collect();

    let total_weight: f64 = devices.iter().map(|d| d.compute_score).sum();

    // Initial share: proportional to compute_score, capped by capacity.
    let mut assigned: Vec<u32> = devices
        .iter()
        .zip(&capacities)
        .map(|(d, &cap)| {
            let share = if total_weight > 0.0 {
                ((d.compute_score / total_weight) * total_layers as f64).floor() as u32
            } else {
                total_layers / devices.len() as u32
            };
            share.min(cap)
        })
        .collect();

    // Distribute the remainder (unassigned layers, from flooring or capacity
    // caps) to devices with spare capacity, ordered fastest-first.
    let mut remaining: u32 = total_layers - assigned.iter().sum::<u32>();

    let mut order: Vec<usize> = (0..devices.len()).collect();
    order.sort_by(|&a, &b| {
        devices[b]
            .compute_score
            .partial_cmp(&devices[a].compute_score)
            .unwrap()
    });

    while remaining > 0 {
        let mut progressed = false;
        for &i in &order {
            if remaining == 0 {
                break;
            }
            if assigned[i] < capacities[i] {
                assigned[i] += 1;
                remaining -= 1;
                progressed = true;
            }
        }
        if !progressed {
            // No device has any spare capacity left — total capacity is less
            // than total_layers. Dump the remainder on the highest-capacity
            // device rather than silently dropping layers; caller (Task 8)
            // is responsible for noticing capacity < total_layers and
            // choosing a smaller model or a lower quantization.
            let biggest = capacities
                .iter()
                .enumerate()
                .max_by_key(|&(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap();
            assigned[biggest] += remaining;
            remaining = 0;
        }
    }

    devices
        .iter()
        .zip(assigned)
        .map(|(d, layer_count)| LayerAssignment {
            device_id: d.device_id.clone(),
            layer_count,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use telemetry::DeviceProfile;

    fn profile(id: &str, score: f64, mem_bytes: u64) -> DeviceProfile {
        DeviceProfile {
            device_id: id.to_string(),
            compute_score: score,
            available_mem_bytes: mem_bytes,
            battery_percent: None,
        }
    }

    #[test]
    fn equal_devices_split_layers_evenly() {
        let devices = vec![
            profile("a", 10.0, 1_000_000_000),
            profile("b", 10.0, 1_000_000_000),
        ];
        let plan = plan_placement(&devices, 32, 1_000_000);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].layer_count, 16);
        assert_eq!(plan[1].layer_count, 16);
    }

    #[test]
    fn faster_device_gets_more_layers() {
        let devices = vec![
            profile("fast", 30.0, 10_000_000_000),
            profile("slow", 10.0, 10_000_000_000),
        ];
        let plan = plan_placement(&devices, 40, 1_000_000);
        // weight ratio 3:1 -> fast should get roughly 3x slow's layers
        assert!(plan[0].layer_count > plan[1].layer_count * 2);
        assert_eq!(plan[0].layer_count + plan[1].layer_count, 40);
    }

    #[test]
    fn memory_constrained_device_spills_remainder_to_others() {
        // "tiny" wants an equal share by compute score but physically cannot
        // hold more than 5 layers; the rest must go to "big".
        let devices = vec![
            profile("tiny", 10.0, 5_000_000), // fits exactly 5 layers at 1MB/layer
            profile("big", 10.0, 1_000_000_000),
        ];
        let plan = plan_placement(&devices, 40, 1_000_000);
        let tiny = plan.iter().find(|p| p.device_id == "tiny").unwrap();
        let big = plan.iter().find(|p| p.device_id == "big").unwrap();
        assert!(tiny.layer_count <= 5);
        assert_eq!(tiny.layer_count + big.layer_count, 40);
    }

    #[test]
    fn all_layers_are_always_assigned() {
        let devices = vec![
            profile("a", 7.0, 2_000_000_000),
            profile("b", 3.0, 2_000_000_000),
            profile("c", 5.0, 2_000_000_000),
        ];
        let plan = plan_placement(&devices, 33, 1_000_000);
        let total: u32 = plan.iter().map(|p| p.layer_count).sum();
        assert_eq!(total, 33);
    }
}
