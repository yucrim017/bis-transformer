# Get ECS task public IP using external data source
data "external" "task_ip" {
    program = ["bash", "-c", <<-EOT
        TASK_ARN=$(aws ecs list-tasks --cluster ${aws_ecs_cluster.mlflow.name} --service-name ${aws_ecs_service.mlflow.name} --region ${var.aws_region} --query 'taskArns[0]' --output text 2>/dev/null || echo "")
        if [ "$TASK_ARN" != "None" ] && [ "$TASK_ARN" != "" ]; then
            ENI_ID=$(aws ecs describe-tasks --cluster ${aws_ecs_cluster.mlflow.name} --tasks $TASK_ARN --region ${var.aws_region} --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text 2>/dev/null || echo "")
            if [ "$ENI_ID" != "None" ] && [ "$ENI_ID" != "" ]; then
                PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --region ${var.aws_region} --query 'NetworkInterfaces[0].Association.PublicIp' --output text 2>/dev/null || echo "")
                if [ "$PUBLIC_IP" != "None" ] && [ "$PUBLIC_IP" != "" ]; then
                    echo "{\"public_ip\":\"$PUBLIC_IP\"}"
                else
                    echo "{\"public_ip\":\"\"}"
                fi
            else
                echo "{\"public_ip\":\"\"}"
            fi
        else
            echo "{\"public_ip\":\"\"}"
        fi
    EOT
    ]
}

output "mlflow_url" {
    description = "MLflow UI URL"
    value = data.external.task_ip.result.public_ip != "" ? "http://${data.external.task_ip.result.public_ip}:5000" : "ECS task not ready yet - check AWS console"
}

output "ecs_cluster_name" {
    description = "ECS Cluster Name"
    value = aws_ecs_cluster.mlflow.name
}

output "ecs_service_name" {
    description = "ECS Service Name"
    value = aws_ecs_service.mlflow.name
}

output "s3_bucket_name" {
    description = "S3 Bucket Name"
    value = data.aws_s3_bucket.mlflow_artifacts.bucket
}