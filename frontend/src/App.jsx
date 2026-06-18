/**
 * frontend/src/App.jsx
 *
 * Main app component. Owns top-level layout and auth state.
 * The search flow is self-contained in SearchBar + RecipeResults
 * via the useSearch hook — App just provides the shell.
 */

import { useState, useEffect } from "react";
import { SearchBar } from "./components/SearchBar";
import { isLoggedIn, logout } from "./api/client";


function App() {
  const [loggedIn, setLoggedIn] = useState(isLoggedIn());

  function handleLogout() {
    logout();
    setLoggedIn(false);
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundImage: "url('/OTL3TX5YGBNCRBRWEFHTQKR6CI.jpg')",
        backgroundRepeat: "no-repeat",
        backgroundSize: "cover",
        backgroundPosition: "center center",
      }}
    >
      {/* Dark overlay — matches original rgba(0,0,0,0.3) */}
      <div style={{ minHeight: "100vh", backgroundColor: "rgba(0,0,0,0.3)" }}>

        {/* Nav */}
        <nav className="navbar navbar-expand-lg navbar-dark" style={{ 
          backgroundColor: "rgba(0,0,0,0.6)",
          position: "sticky",
          top: 0,
          zIndex: 1000,
          backdropFilter: "blur(4px)", 
        }}>
          <div className="container">
            <a className="navbar-brand d-flex align-items-center" href="/">
              <img
                src="/mealeon_no_bg.png"
                alt="MeaLeon"
                height="32"
                className="me-2"
                onError={(e) => { e.target.style.display = "none"; }}
              />
              MeaLeon
            </a>
            <div className="ms-auto d-flex gap-2">
              {loggedIn ? (
                <>
                  <a className="btn btn-outline-light btn-sm" href="/auth/login">
                    Profile
                  </a>
                  <button
                    className="btn btn-outline-light btn-sm"
                    onClick={handleLogout}
                  >
                    Log out
                  </button>
                </>
              ) : (
                <>
                  <a className="btn btn-outline-light btn-sm" href="/auth/login">
                    Log in
                  </a>
                  <a className="btn btn-light btn-sm" href="/auth/register">
                    Sign up
                  </a>
                </>
              )}
            </div>
          </div>
        </nav>

        {/* Jumbotron — matches original style */}
        <div className="container mt-5">
          <div
            className="p-4 p-md-5 mb-4 rounded text-white"
            style={{ backgroundColor: "rgba(0,0,0,0.4)" }}
          >
            <div className="text-center mb-4">
              <img
                src="/mealeon_no_bg.png"
                alt="MeaLeon mascot"
                style={{ maxHeight: 120 }}
                onError={(e) => { e.target.style.display = "none"; }}
              />
              <h1 className="display-5 fw-bold mt-2">MeaLeon</h1>
              <p className="lead">
                Enter a recipe you like and the cuisine it's from — get 5 similar
                recipes from other cuisines!
              </p>
            </div>
            <SearchBar />
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center text-white py-3" style={{ backgroundColor: "rgba(0,0,0,0.3)" }}>
          <small>
            MeaLeon &mdash; similar recipes, different cuisines &mdash;{" "}
            <a
              href="https://developer.edamam.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-white"
            >
              Powered by Edamam
            </a>
          </small>
        </footer>

      </div>
    </div>
  );
}

export default App;
