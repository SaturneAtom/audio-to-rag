#!/usr/bin/env bash
# Creates (or updates) the disposable test environment and the GitHub deploy role,
# then prints the repository variables the Deploy workflow needs.
# Run it where the AWS CLI has admin-level credentials, for example AWS CloudShell:
#   bash create-test-environment.sh
# Remove everything later with delete-test-environment.sh.
set -euo pipefail

REGION="${REGION:-us-east-1}"
NAME="${NAME:-audio-to-rag-test}"
GITHUB_OWNER="${GITHUB_OWNER:-SaturneAtom}"
GITHUB_REPO="${GITHUB_REPO:-audio-to-rag}"
HERE="$(cd "$(dirname "$0")" && pwd)"

output() {
  aws cloudformation describe-stacks --region "$REGION" --stack-name "$1" \
    --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text
}

echo "== Test environment stack: $NAME ($REGION)"
aws cloudformation deploy --region "$REGION" --stack-name "$NAME" \
  --template-file "$HERE/test-environment.yml" \
  --capabilities CAPABILITY_NAMED_IAM --no-fail-on-empty-changeset

if aws iam list-open-id-connect-providers --query 'OpenIDConnectProviderList[].Arn' --output text \
    | grep -q 'token.actions.githubusercontent.com'; then
  CREATE_OIDC=false
else
  CREATE_OIDC=true
fi

echo "== GitHub deploy role stack: $NAME-github-deploy (create OIDC provider: $CREATE_OIDC)"
aws cloudformation deploy --region "$REGION" --stack-name "$NAME-github-deploy" \
  --template-file "$HERE/github-actions-role.yml" \
  --capabilities CAPABILITY_NAMED_IAM --no-fail-on-empty-changeset \
  --parameter-overrides \
    GitHubOwner="$GITHUB_OWNER" GitHubRepo="$GITHUB_REPO" \
    EcrRepositoryName="$(output "$NAME" RepositoryName)" \
    TaskRoleArn="$(output "$NAME" TaskRoleArn)" \
    ExecutionRoleArn="$(output "$NAME" ExecutionRoleArn)" \
    CreateOidcProvider="$CREATE_OIDC"

echo
echo "== Repository variables"
echo "AWS_ROLE_ARN=$(output "$NAME-github-deploy" RoleArn)"
echo "AWS_REGION=$REGION"
echo "ECR_REPOSITORY=$(output "$NAME" RepositoryName)"
echo "ECS_TASK_FAMILY=$(output "$NAME" TaskFamily)"
echo "ECS_CONTAINER_NAME=$(output "$NAME" ContainerName)"
echo "SETUP_DONE"
