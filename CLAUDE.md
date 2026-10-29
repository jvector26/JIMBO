# JIMBO: start here
Everything lives in this repo (no handoff zip since 2026-10-29). New chat:
1. Clone to /home/claude/jimbo (add_repo jvector26/JIMBO, access push), then `bash /home/claude/jimbo/work/setup.sh`.
2. Read /home/claude/HANDOFF.md (= work/HANDOFF.md): state, user's working style, production chain, research bar, backlog.
3. Work from /home/claude as before. Commit sandbox changes (work/) to main with everything else; `git pull --rebase` first
   (the nightly bot commits several times a day). End-of-session housekeeping = update work/HANDOFF.md (+ RESEARCH_LOG.md) and push.
Layout: pipeline/ model/ live/ state/ = nightly run on GitHub Actions; site/ = phone app (built by work/code/build_app.py);
work/ = research + preseason model code, data, dashboard template (code/dash_v2.html), handoff notes.
