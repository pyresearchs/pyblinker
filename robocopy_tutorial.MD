
# Tutorial: Copy 3TB External (E:) → 2TB Mirror (D:) with integrity checks (Windows 11)

## A) One-time setup (Storage Spaces mirror)

1. **Control Panel → System and Security → Storage Spaces**
2. **Create a new pool and storage space**
3. Select **ONLY the two 2TB drives**
4. Resiliency: **Two-way mirror**
5. Size: set about **1.81 TB**
6. Finish → You’ll get a drive letter (example **D:**)

✅ Your 3TB external stays separate (example **E:**).

---

## B) Identify your drive letters every time

* External 3TB (Orico): usually **E:**
* Mirrored space: usually **D:**

Open **File Explorer** to confirm.

---

## C) Copy E:\ → D:\ (root-to-root)

### 1) Copy command (safe + fast, excludes system folders)

Open **Windows Terminal / CMD** and run:

```bat
robocopy E:\ D:\ /E /COPY:DAT /DCOPY:DAT /R:0 /W:0 /Z /MT:32 /XJ /XD "E:\System Volume Information" "E:\$RECYCLE.BIN" /TEE /LOG:"C:\Users\balan\OneDrive\Desktop\E_to_D_copy.log"
```

**What it does**

* Copies *everything* from **E:** into **D:** (folders directly under D:)
* Skips locked Windows folders that cause “Access denied”
* Writes a log to your OneDrive Desktop

### 2) If you accidentally close CMD

Just run the same command again — Robocopy will **continue** and skip what’s already copied.

---

## D) Verify (integrity check pass)

After the copy finishes, run a dry-run verify:

```bat
robocopy E:\ D:\ /E /L /R:0 /W:0 /NJH /NJS /NP /XD "E:\System Volume Information" "E:\$RECYCLE.BIN" /LOG:"C:\Users\balan\OneDrive\Desktop\E_to_D_verify.log"
```

✅ If it would copy **0 files** and shows **FAILED: 0**, you’re good.

---

## E) Check Robocopy result code (quick health check)

Right after the copy, run:

```bat
echo %errorlevel%
```

* **0–3** = normal/OK
* **8+** = issues (failed copies)

---

# Tutorial: Use E:\ as BACKUP for D:\ (update the backup later)

This copies new/changed stuff from **D:\ → E:** (does **not** delete extra files on E:).

```bat
robocopy D:\ E:\ /E /COPY:DAT /DCOPY:DAT /R:0 /W:0 /MT:32 /XJ /XD "D:\System Volume Information" "D:\$RECYCLE.BIN" /TEE /LOG:"C:\Users\balan\OneDrive\Desktop\D_to_E_backup.log"
```

### Verify the backup

```bat
robocopy D:\ E:\ /E /L /R:0 /W:0 /NJH /NJS /NP /XD "D:\System Volume Information" "D:\$RECYCLE.BIN" /LOG:"C:\Users\balan\OneDrive\Desktop\D_to_E_verify.log"
```

---

## Optional: “Exact mirror backup” (⚠️ can delete files)

Only if you want E:\ to become EXACTLY the same as D::

```bat
robocopy D:\ E:\ /MIR /COPY:DAT /DCOPY:DAT /R:0 /W:0 /MT:32 /XJ /XD "D:\System Volume Information" "D:\$RECYCLE.BIN" /TEE /LOG:"C:\Users\balan\OneDrive\Desktop\D_to_E_MIRROR.log"
```

⚠️ `/MIR` will delete from **E:** anything that isn’t on **D:**.

---

## Tips to make it fast and stable

* Use a **rear motherboard USB 3.x** port (avoid hubs/front ports)
* Keep PC awake during copy
* If USB is stable, you can remove `/Z` for slightly more speed

