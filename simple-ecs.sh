#!/bin/bash
# Script ECS simplifié - Version 10 lignes au lieu de 119 !
exec > /var/log/ecs-simple.log 2>&1

echo "=== DÉBUT ECS SIMPLE $(date) ==="

# Configuration ECS (suffit avec une AMI ECS optimisée)
echo "ECS_CLUSTER=${ECS_CLUSTER_NAME:-your-ecs-cluster-name}" >> /etc/ecs/ecs.config
echo "ECS_ENABLE_GPU_SUPPORT=true" >> /etc/ecs/ecs.config

# Restart ECS agent
systemctl restart ecs

echo "=== FIN - ECS prêt $(date) ===" 