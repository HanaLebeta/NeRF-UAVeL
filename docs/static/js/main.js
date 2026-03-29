/**
 * NeRF-UAVeL Project Page - Main JavaScript
 */

document.addEventListener('DOMContentLoaded', function () {

  // --- Navbar scroll effect ---
  const navbar = document.getElementById('navbar');
  function handleNavbarScroll() {
    if (window.scrollY > 40) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
  }
  window.addEventListener('scroll', handleNavbarScroll, { passive: true });
  handleNavbarScroll();

  // --- Mobile navigation toggle ---
  const navToggle = document.getElementById('nav-toggle');
  const navLinks = document.getElementById('nav-links');

  if (navToggle && navLinks) {
    navToggle.addEventListener('click', function () {
      navLinks.classList.toggle('active');
      navToggle.classList.toggle('active');
    });

    // Close mobile nav on link click
    navLinks.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        navLinks.classList.remove('active');
        navToggle.classList.remove('active');
      });
    });
  }

  // --- Smooth scroll for anchor links ---
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      var targetId = this.getAttribute('href');
      if (targetId === '#') return;
      var target = document.querySelector(targetId);
      if (target) {
        e.preventDefault();
        var navHeight = navbar ? navbar.offsetHeight : 0;
        var targetPos = target.getBoundingClientRect().top + window.pageYOffset - navHeight;
        window.scrollTo({
          top: targetPos,
          behavior: 'smooth'
        });
      }
    });
  });

  // --- BibTeX copy button ---
  var copyBtn = document.getElementById('copy-bibtex');
  var bibtexContent = document.getElementById('bibtex-content');

  if (copyBtn && bibtexContent) {
    copyBtn.addEventListener('click', function () {
      var text = bibtexContent.textContent || bibtexContent.innerText;

      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          showCopied();
        }).catch(function () {
          fallbackCopy(text);
        });
      } else {
        fallbackCopy(text);
      }
    });

    function fallbackCopy(text) {
      var textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.style.position = 'fixed';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.select();
      try {
        document.execCommand('copy');
        showCopied();
      } catch (err) {
        // silent fail
      }
      document.body.removeChild(textarea);
    }

    function showCopied() {
      copyBtn.classList.add('copied');
      copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
      setTimeout(function () {
        copyBtn.classList.remove('copied');
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
      }, 2000);
    }
  }

  // --- Scroll-triggered fade-in animations ---
  var animateElements = document.querySelectorAll(
    '.module-card, .qual-item, .table-wrapper, .improvement-callout, .abstract-content, .architecture-figure, .bibtex-wrapper'
  );

  if ('IntersectionObserver' in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('fade-in-visible');
          observer.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -40px 0px'
    });

    animateElements.forEach(function (el) {
      el.classList.add('fade-in');
      observer.observe(el);
    });
  }

  // Add CSS for fade-in animation dynamically
  var style = document.createElement('style');
  style.textContent = [
    '.fade-in {',
    '  opacity: 0;',
    '  transform: translateY(24px);',
    '  transition: opacity 0.6s cubic-bezier(0.4, 0, 0.2, 1), transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);',
    '}',
    '.fade-in-visible {',
    '  opacity: 1;',
    '  transform: translateY(0);',
    '}'
  ].join('\n');
  document.head.appendChild(style);

  // --- Active nav link highlighting ---
  var sections = document.querySelectorAll('section[id]');
  function highlightNav() {
    var scrollPos = window.scrollY + 100;
    sections.forEach(function (section) {
      var top = section.offsetTop;
      var height = section.offsetHeight;
      var id = section.getAttribute('id');
      var link = document.querySelector('.nav-links a[href="#' + id + '"]');
      if (link) {
        if (scrollPos >= top && scrollPos < top + height) {
          link.style.color = '#1d4ed8';
          link.style.background = '#eef2ff';
        } else {
          link.style.color = '';
          link.style.background = '';
        }
      }
    });
  }
  window.addEventListener('scroll', highlightNav, { passive: true });
  highlightNav();

});
