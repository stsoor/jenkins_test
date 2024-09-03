# Use the official Jenkins LTS with JDK 17 image as a base
FROM jenkins/jenkins:lts-jdk17

# Switch to the root user to install Docker
USER root

# Install Docker
RUN apt-get update && \
    apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add - && \
    echo "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce-cli && \
    rm -rf /var/lib/apt/lists/*

# Add the Jenkins user to the Docker group
RUN groupadd docker && usermod -aG docker jenkins

# Switch back to the jenkins user
USER jenkins

# The default command provided by the base image will be preserved
CMD ["jenkins.sh"]