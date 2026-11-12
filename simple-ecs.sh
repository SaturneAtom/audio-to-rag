#!/bin/bash
# Minimal ECS agent configuration for a GPU EC2 instance (ECS-optimized AMI).
exec > /var/log/ecs-simple.log 2>&1

echo "=== ECS setup start $(date) ==="

# Register the instance with the cluster and enable GPU support
echo "ECS_CLUSTER=${ECS_CLUSTER_NAME:-your-ecs-cluster-name}" >> /etc/ecs/ecs.config
echo "ECS_ENABLE_GPU_SUPPORT=true" >> /etc/ecs/ecs.config

# Restart the ECS agent
systemctl restart ecs

echo "=== ECS ready $(date) ==="
