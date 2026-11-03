-- LexTechSolver — Supabase schema
-- Run once in: Supabase dashboard → SQL Editor → New query → paste → Run

-- ── Users ────────────────────────────────────────────────────────────────────
create table if not exists public.users (
    id            uuid primary key default gen_random_uuid(),
    name          text        not null,
    email         text        not null unique,
    password_hash text        not null,                 -- bcrypt hash, never plain text
    api_key       text        not null unique
                  default ('lxt-' || replace(gen_random_uuid()::text, '-', '')),
    plan          text        not null default 'standard',
    active        boolean     not null default true,    -- flip to false to block instantly
    expires_at    timestamptz,                          -- subscription end date (null = no expiry)
    solves_total  bigint      not null default 0,
    last_solve_at timestamptz,
    last_ip       text,
    created_at    timestamptz not null default now()
);

-- ── Usage log (one row per solve request) ────────────────────────────────────
create table if not exists public.usage_log (
    id            bigserial primary key,
    user_id       uuid        not null references public.users(id) on delete cascade,
    ip            text,
    success       boolean     not null default true,
    processing_ms integer,
    created_at    timestamptz not null default now()
);

create index if not exists usage_log_user_time_idx
    on public.usage_log (user_id, created_at desc);

-- ── Atomic solve counter ─────────────────────────────────────────────────────
-- Increments in a single statement so concurrent workers never lose a count
create or replace function public.record_solve(p_user_id uuid, p_ip text)
returns void
language sql
security definer
set search_path = public
as $$
    update public.users
       set solves_total  = solves_total + 1,
           last_solve_at = now(),
           last_ip       = p_ip
     where id = p_user_id;
$$;

-- Only the backend (service_role) may call it
revoke execute on function public.record_solve(uuid, text) from public, anon, authenticated;

-- ── Lock the tables down ─────────────────────────────────────────────────────
-- RLS on + no policies = the public anon key can read/write nothing.
-- Your backend uses the service_role key, which bypasses RLS.
alter table public.users     enable row level security;
alter table public.usage_log enable row level security;
