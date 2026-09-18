import { PGlite } from '@electric-sql/pglite';
import fs from 'node:fs';
import assert from 'node:assert/strict';
const pg=new PGlite();
await pg.exec(`create schema auth; create role anon; create role authenticated; create role service_role bypassrls;
create function auth.uid() returns uuid language sql as $$ select nullif(current_setting('request.jwt.claim.sub', true),'')::uuid $$;
grant usage on schema auth to authenticated;
create table public.entitlements(user_id uuid primary key,is_pro boolean,expires_at timestamptz,source text);
alter table public.entitlements enable row level security;
grant all on public.entitlements to anon,authenticated,service_role;
create policy "Service role can manage entitlements" on entitlements for all to public using(true) with check(true);
create policy "Users can view own entitlements" on entitlements for select to public using(auth.uid()=user_id);
insert into entitlements values('11111111-1111-4111-8111-111111111111',true,null,'promo'),('22222222-2222-4222-8222-222222222222',false,null,'stripe');`);
const sql=fs.readFileSync(new URL('../migrations/013_entitlement_access.sql',import.meta.url),'utf8');await pg.exec(sql);await pg.exec(sql);
let checks=0;const eq=(a,b)=>{assert.deepEqual(a,b);checks++;};
await pg.exec("set role authenticated; set request.jwt.claim.sub='11111111-1111-4111-8111-111111111111';");
eq((await pg.query('select * from entitlements')).rows.length,1);
eq((await pg.query('select is_pro from entitlements')).rows[0].is_pro,true);
for(const statement of ["update entitlements set is_pro=false", "delete from entitlements", "truncate entitlements", "insert into entitlements values(gen_random_uuid(),true,null,'promo')"]){
 let denied=false;try{await pg.exec(statement);}catch{denied=true;}eq(denied,true);
}
await pg.exec('reset role; set role anon');
let denied=false;try{await pg.query('select * from entitlements');}catch{denied=true;}eq(denied,true);
await pg.exec('reset role; set role service_role');eq((await pg.query('select * from entitlements')).rows.length,2);
await pg.exec("update entitlements set is_pro=true where source='stripe'");
eq((await pg.query('select count(*)::integer as n from entitlements where is_pro')).rows[0].n,2);
console.log(`${checks} entitlement access assertions passed; migration reapplied successfully.`);await pg.close();
