#!/usr/bin/env bash
# Deletes everything created by create-test-environment.sh and by the Deploy workflow:
# - stops any running worker task,
# - empties the S3 bucket,
# - deletes both stacks (CloudFormation empties the ECR repository and removes the
#   queues, tables, log group, cluster and roles; the GitHub OIDC provider is removed
#   only if the role stack created it),
# - deregisters and deletes every task definition revision of the family, including
#   the ones the Deploy workflow registered outside CloudFormation.
# Ends with a check that lists anything still named after the environment.
set -euo pipefail

REGION="${REGION:-us-east-1}"
NAME="${NAME:-audio-to-rag-test}"

if aws ecs describe-clusters --region "$REGION" --clusters "$NAME" \
    --query 'clusters[?status==`ACTIVE`].clusterName' --output text | grep -q .; then
  for TASK in $(aws ecs list-tasks --region "$REGION" --cluster "$NAME" --query 'taskArns' --output text); do
    echo "== Stopping task $TASK"
    aws ecs stop-task --region "$REGION" --cluster "$NAME" --task "$TASK" >/dev/null
  done
fi

BUCKET=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='BucketName'].OutputValue" --output text 2>/dev/null || true)
if [ -n "$BUCKET" ] && [ "$BUCKET" != "None" ]; then
  echo "== Emptying s3://$BUCKET"
  aws s3 rm "s3://$BUCKET" --recursive --region "$REGION"
fi

for STACK in "$NAME-github-deploy" "$NAME"; do
  if aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" >/dev/null 2>&1; then
    echo "== Deleting stack $STACK"
    aws cloudformation delete-stack --region "$REGION" --stack-name "$STACK"
    aws cloudformation wait stack-delete-complete --region "$REGION" --stack-name "$STACK"
  fi
done

echo "== Task definition revisions of $NAME"
for STATUS in ACTIVE INACTIVE; do
  for ARN in $(aws ecs list-task-definitions --region "$REGION" --family-prefix "$NAME" \
      --status "$STATUS" --query 'taskDefinitionArns' --output text); do
    [ "$STATUS" = ACTIVE ] && aws ecs deregister-task-definition --region "$REGION" \
      --task-definition "$ARN" >/dev/null
    aws ecs delete-task-definitions --region "$REGION" --task-definitions "$ARN" >/dev/null
    echo "   deleted $ARN"
  done
done

echo "== Leftover check (should print nothing below)"
aws cloudformation describe-stacks --region "$REGION" --stack-name "$NAME" >/dev/null 2>&1 && echo "stack $NAME"
aws ecr describe-repositories --region "$REGION" --repository-names "$NAME" >/dev/null 2>&1 && echo "ECR $NAME"
aws sqs get-queue-url --region "$REGION" --queue-name "$NAME" >/dev/null 2>&1 && echo "SQS $NAME"
aws dynamodb list-tables --region "$REGION" --query "TableNames[?starts_with(@, '$NAME')]" --output text
aws s3api list-buckets --query "Buckets[?starts_with(Name, '$NAME')].Name" --output text
aws iam list-roles --query "Roles[?starts_with(RoleName, '$NAME') || RoleName=='audio-to-rag-github-deploy'].RoleName" --output text
aws ecs list-task-definitions --region "$REGION" --family-prefix "$NAME" --query 'taskDefinitionArns' --output text
echo "DELETE_DONE"
