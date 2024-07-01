--
-- PostgreSQL database dump
--

-- Dumped from database version 16.3 (Debian 16.3-1.pgdg120+1)
-- Dumped by pg_dump version 16.3 (Debian 16.3-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: uuid-ossp; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;


--
-- Name: EXTENSION "uuid-ossp"; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION "uuid-ossp" IS 'generate universally unique identifiers (UUIDs)';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: documents; Type: TABLE; Schema: public; Owner: user
--

CREATE TABLE public.documents (
    id uuid DEFAULT public.uuid_generate_v4() NOT NULL,
    url text,
    title text,
    headings text[],
    page_text text,
    keywords text[],
    accessed_timestamp timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    internal_links text[],
    external_links text[]
);


ALTER TABLE public.documents OWNER TO "user";

--
-- Data for Name: documents; Type: TABLE DATA; Schema: public; Owner: user
--

COPY public.documents (id, url, title, headings, page_text, keywords, accessed_timestamp, internal_links, external_links) FROM stdin;
026e6158-f106-4e53-ac96-cbcffe60119c	url1	Title 1	{heading1}	page text 1	{keyword1,keyword2,keyword3,keyword4,keyword5,keyword6,keyword7}	2024-07-01 19:36:27.382639	{}	{myfirstevencoolerexternallink1,myfirstevencoolerexternallink2}
1557ee81-c41c-4c1e-8d34-e2b5f26fab93	url2	\N	{"",NULL}		{NULL}	2024-06-30 19:36:27.382559	{NULL}	{}
\.


--
-- Name: documents documents_pkey; Type: CONSTRAINT; Schema: public; Owner: user
--

ALTER TABLE ONLY public.documents
    ADD CONSTRAINT documents_pkey PRIMARY KEY (id);


--
-- PostgreSQL database dump complete
--

