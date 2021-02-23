#include "GLFramebuffer.h"

// GL
#include "GL/glew.h"
#include "glfw/glfw3.h"
#include "glfw/glfw3native.h"

// C++
#include <assert.h>

namespace Tracer
{
	GLFramebuffer::GLFramebuffer(const int2& resolution)
	{
		mGlTexture = new GLTexture(resolution, GLTexture::Types::Byte4);

		// create the framebuffer
		glGenFramebuffers(1, &mId);
		glBindFramebuffer(GL_FRAMEBUFFER, mId);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, mGlTexture->ID(), 0);

		mDrawBuffer = { GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(1, &mDrawBuffer);

		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}



	GLFramebuffer::~GLFramebuffer()
	{
		delete mGlTexture;
		glDeleteFramebuffers(1, &mId);
	}



	void GLFramebuffer::Bind() const
	{
		glBindFramebuffer(GL_FRAMEBUFFER, mId);
	}



	void GLFramebuffer::Unbind() const
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}
