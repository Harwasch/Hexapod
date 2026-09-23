-- Creates the database used by the backend test-suite alongside the dev database.
CREATE DATABASE twin_test OWNER twin;
\connect twin_test
CREATE EXTENSION IF NOT EXISTS postgis;
\connect twin
CREATE EXTENSION IF NOT EXISTS postgis;
