#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <clocale>
#include <malloc.h>

const unsigned int window_width = 512;
const unsigned int window_height = 512;

void initGL();

GLuint *bufferID;

void initBuffers(GLuint*&);
void transformBuffers(GLuint*);
void outputBuffers(GLuint*);

int main() {
	setlocale(LC_CTYPE, "rus");

	initGL();
	
	bufferID = (GLuint*)calloc(2, sizeof(GLuint));
	
	initBuffers(bufferID);
	transformBuffers(bufferID);
	outputBuffers(bufferID);

	glDeleteBuffers(2, bufferID);
	free(bufferID);
	glfwTerminate();

	return 0;
}

void initGL() {
	GLFWwindow *window;
	
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		getchar();
		return;
	
	}

	glfwWindowHint(GLFW_VISIBLE, 0);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	window = glfwCreateWindow(window_width, window_height, "Template window", NULL, NULL);

	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. \n");
		getchar();
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		getchar();
		glfwTerminate();
		return;
	}
}
