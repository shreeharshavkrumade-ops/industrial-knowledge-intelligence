import sys; sys.path.insert(0, '.')
from src.auth import init_db, register_user, login_user, validate_session, logout

init_db()
print("=== SMOKE TEST ===")

r = register_user("smoketest_admin", "admin@gmail.com", "Adm!n_Str0ngPass2024")
print("Register:", r)

r2 = login_user("smoketest_admin", "Adm!n_Str0ngPass2024")
print("Login username ok:", r2["ok"], "user:", r2.get("username"))
tok = r2["token"]

r3 = login_user("admin@gmail.com", "Adm!n_Str0ngPass2024")
print("Login email ok:", r3["ok"])

v = validate_session(tok)
print("Session valid:", v["username"] if v else None)

r4 = login_user("smoketest_admin", "WrongPassword12345!")
print("Wrong pw rejected:", not r4["ok"], "msg:", r4["error"][:50])

r5 = login_user("nobody_here", "WrongPassword12345!")
print("Unknown user rejected:", not r5["ok"], "msg:", r5["error"][:50])

logout(tok)
v2 = validate_session(tok)
print("After logout session None:", v2 is None)

print("=== ALL OK ===")
