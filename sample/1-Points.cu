// Compilation: nvcc 1-Points.cu -o out -lglfw -lGL -lGLEW -lglut -lGLU; ./out
// Verbatim: https://cs.lmu.edu/~ray/notes/openglexamples/ 
// Shows a trivial use of OpenGL bitmaps, so trivial in fact, that it uses
// nothing but the default settings for glPixelStore*.
#include <random>
#include <iostream>
#ifdef __APPLE_CC__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <cstdlib>

// A fish bitmap, size is 27x11, but all rows must have a multiple of 8 bits,
// so we define it like it is 32x11.
GLubyte fish[] = {
  0xFF, 0x60, 0x01, 0x00,
  0x00, 0x90, 0x01, 0x00,
  0x03, 0xf8, 0x02, 0x80,
  0x1c, 0x37, 0xe4, 0x40,
  0x20, 0x40, 0x90, 0x40,
  0xc0, 0x40, 0x78, 0x80,
  0x41, 0x37, 0x84, 0x80,
  0x1c, 0x1a, 0x04, 0x80,
  0x03, 0xe2, 0x02, 0x40,
  0x00, 0x11, 0x01, 0x40,
  0x00, 0x0f, 0x00, 0xe0,
};

// Return a random float in the range 0.0 to 1.0.
GLfloat randomFloat() {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<> dist(0, 1);
  return (GLfloat)dist(rng);
}

// On reshape, uses an orthographic projection with world coordinates ranging
// from 0 to 1 in the x and y directions, and -1 to 1 in z.
void reshape(int width, int height) {
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, 1, 0, 1);
}

// Clears the window then plots 20 fish bitmaps in random colors at random
// positions.
void display() {
  glClear(GL_COLOR_BUFFER_BIT);
  for (int i = 0; i < 20; i++) {
    glColor3f(randomFloat(), randomFloat(), randomFloat());
    glRasterPos3f(randomFloat(), randomFloat(), 0.0);
    glBitmap(27, 11, 0, 0, 0, 0, fish);
  }
  glFlush();
}

// The usual main() for a GLUT application.
int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
  glutInitWindowSize(400, 300);
  glutCreateWindow("Pixels");
  glutReshapeFunc(reshape);
  glutDisplayFunc(display);
  glutMainLoop();
}