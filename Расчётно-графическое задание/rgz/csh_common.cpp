#include <GL/glew.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <time.h>

void checkErrors(std::string desc) {
	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		fprintf(stderr, "OpenGL error in \"%s\": %s (%d)\n", desc.c_str(), gluErrorString(e), e);
		getchar();
		exit(20);
	}
}

const int N = (1 << 8);
GLuint genInitProg();
GLuint genTransformProg();

int initBuffers(GLuint*& bufferID) {
	glGenBuffers(2, bufferID);
	
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID[0]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, N * sizeof(float), 0, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID[1]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, N * sizeof(float), 0, GL_DYNAMIC_DRAW);
	
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID[0]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferID[1]);

	GLuint csInitID = genInitProg();
	glUseProgram(csInitID);
	
	glDispatchCompute(N / 256, 1, 1);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	
	glDeleteProgram(csInitID);	
}

GLuint genInitProg() {
	GLuint progHandle = glCreateProgram();
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
	
	const char *cpSrc[] = {
		"#version 430\n",
		"layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\
		 layout(std430, binding = 0) buffer BufferA{float A[];};\
		 layout(std430, binding = 1) buffer BufferB{float B[];};\
		 void main() {\
		 	uint index = gl_GlobalInvocationID.x;\
			A[index] = float(index);\
			B[index] = 0.87;\
		}"
	};
	
	glShaderSource(cs, 2, cpSrc, NULL);
	
	glCompileShader(cs);
	int rvalue;
	glGetShaderiv(cs, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in compiling cs\n");
		getchar();
		exit(30);
	}
	glAttachShader(progHandle, cs);

	glLinkProgram(progHandle);
	glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in linking cs\n");
		getchar();
		exit(32);
	}
	checkErrors("Render shaders");

	return progHandle;
}

int transformBuffers(GLuint *bufferID) {
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferID[0]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferID[1]);
	GLuint csTransformID = genTransformProg();

	GLuint query;
	GLuint elapsedTime;
	
	glGenQueries(1, &query);
	glBeginQuery(GL_TIME_ELAPSED, query);
	
	glUseProgram(csTransformID);

	GLuint alphaID = glGetUniformLocation(csTransformID, "alpha");
	float a = 3.0f;
	glUniform1f(alphaID, a);

	glDispatchCompute(N / 256, 1, 1);	
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
	
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectuiv(query, GL_QUERY_RESULT, &elapsedTime);

	printf("OpenGL Time: %f ms\n", elapsedTime / 1000000.0);
	
	glDeleteProgram(csTransformID);	
}

GLuint genTransformProg() {
	GLuint progHandle = glCreateProgram();
	GLuint cs = glCreateShader(GL_COMPUTE_SHADER);

	const char *cpSrc[] = {
		"#version 430\n",
		"layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\
		 layout(std430, binding = 0) buffer BufferA{float A[];};\
		 layout(std430, binding = 1) buffer BufferB{float B[];};\
		 uniform float alpha;\
		 void main() {\
		 	uint index = gl_GlobalInvocationID.x;\
			B[index] = alpha * A[index] + B[index];\
		}"
	};
	
	glShaderSource(cs, 2, cpSrc, NULL);
	
	glCompileShader(cs);
	int rvalue;
	glGetShaderiv(cs, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in compiling cs\n");
		getchar();
		exit(30);
	}
	glAttachShader(progHandle, cs);

	glLinkProgram(progHandle);
	glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in linking cs\n");
		getchar();
		exit(32);
	}
	checkErrors("Render shaders");

	return progHandle;
}

void outputBuffers(GLuint *bufferID) {
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferID[1]);
	float *data = (float*)glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	
	float *hdata = (float*)calloc(N, sizeof(float));
	memcpy(&hdata[0], data, sizeof(float) * N);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	
//	for(int i = 0; i < (1 << 5); i++) {
//		fprintf(stdout, "%g\t", hdata[i]);
//	}
}
